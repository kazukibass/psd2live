package io.github.psd2live.ui

import io.github.psd2live.core.RigEditOverlay
import io.github.psd2live.core.RigKeyformGeometryEdit
import io.github.psd2live.core.RigKeyformSetEdit
import io.github.psd2live.core.RigPreviewModel
import io.github.psd2live.core.RigTargetKind
import io.github.psd2live.core.RigTargetRef
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.umamo.runtime.model.Deformer
import org.umamo.runtime.model.DeformerId
import org.umamo.runtime.model.Drawable
import org.umamo.runtime.model.ParameterId
import java.io.File
import kotlin.math.abs
import kotlin.math.floor

/**
 * Replaces the generated 9-axis head keyforms with measured targets.
 *
 * 1. Every deformer under the head container gets its rest lattice at every ParamAngleX / ParamAngleY
 *    key, so the generated head-turn motion no longer applies.
 * 2. Every head drawable gets vertex keyforms on ParamAngleX x ParamAngleY: at each key the world
 *    positions are pushed by a dense displacement field (tools/angle_targets.py), and the push is
 *    converted back into the drawable's local space. Existing keyforms (eye open, mouth form/open,
 *    brow, iris form) are kept and combined, so blinking and lip-sync still work in every direction.
 */
object HeadRetarget {
	private const val HEAD_ROOT = "DeformHeadContainer"
	private val ANGLE_PARAMS = listOf("ParamAngleX", "ParamAngleY")

	private class Field(val ax: Float, val ay: Float, val x0: Float, val y0: Float, val step: Float,
		val nx: Int, val ny: Int, val dx: FloatArray, val dy: FloatArray) {
		fun sample(x: Float, y: Float): Pair<Float, Float> {
			val fx = ((x - x0) / step).coerceIn(0f, nx - 1.001f)
			val fy = ((y - y0) / step).coerceIn(0f, ny - 1.001f)
			val i = floor(fx).toInt()
			val j = floor(fy).toInt()
			val tx = fx - i
			val ty = fy - j
			fun at(a: FloatArray, ii: Int, jj: Int) = a[jj * nx + ii]
			fun lerp(a: FloatArray) = (at(a, i, j) * (1 - tx) + at(a, i + 1, j) * tx) * (1 - ty) +
				(at(a, i, j + 1) * (1 - tx) + at(a, i + 1, j + 1) * tx) * ty
			return lerp(dx) to lerp(dy)
		}
	}

	private fun loadFields(path: File): List<Field> =
		Json.parseToJsonElement(path.readText()).jsonObject.getValue("poses").jsonArray.map { e ->
			val o = e.jsonObject
			fun f(k: String) = o.getValue(k).jsonPrimitive.float
			fun arr(k: String) = o.getValue(k).jsonArray.map { it.jsonPrimitive.float }.toFloatArray()
			Field(f("angleX"), f("angleY"), f("x0"), f("y0"), f("step"), o.getValue("nx").jsonPrimitive.int,
				o.getValue("ny").jsonPrimitive.int, arr("dx"), arr("dy"))
		}

	fun build(model: RigPreviewModel, fieldsPath: File): RigEditOverlay {
		val puppet = model.rig.puppet
		val fields = loadFields(fieldsPath)
		val byId = puppet.deformers.associateBy { it.id }
		fun underHead(id: DeformerId?): Boolean {
			var cur = id
			while (cur != null) {
				if (cur.raw == HEAD_ROOT) return true
				cur = byId[cur]?.parent
			}
			return false
		}

		// Deleting both angle axes would leave a deformer without any form (and drop its children), so
		// every angle keyform is overwritten with the deformer's rest lattice instead.
		val neutralized = mutableListOf<RigKeyformSetEdit>()
		for (d in puppet.deformers.filter { underHead(it.id) }) {
			val grid = (d as? Deformer.Warp)?.geometryGrid ?: continue
			val angleAxes = grid.axes.indices.filter { grid.axes[it].parameterId.raw in ANGLE_PARAMS }
			if (angleAxes.isEmpty()) continue
			fun atRest(c: IntArray) = IntArray(c.size) { i ->
				if (i in angleAxes) grid.axes[i].keys.indexOfFirst { abs(it) < 1e-4 } else c[i]
			}
			for (cell in grid.cells) {
				val restForm = grid.cells.first { it.coordinate.contentEquals(atRest(cell.coordinate)) }.form
				val coord = grid.axes.mapIndexed { i, axis -> axis.parameterId.raw to axis.keys[cell.coordinate[i]] }.toMap()
				neutralized += RigKeyformSetEdit(RigTargetRef(RigTargetKind.WARP_DEFORMER, d.id.raw), coord,
					RigKeyformGeometryEdit(controlPoints = restForm.controlPoints.toList()))
			}
		}
		val sets = mutableListOf<RigKeyformSetEdit>()
		val rest = RigCanvasSupport.evaluate(model, emptyMap())
		for (drawable in puppet.drawables.filter { underHead(it.parentDeformerId) }) {
			val mesh = drawable.mesh ?: continue
			val n = mesh.positions.size / 2
			val world = rest.worldPositions[drawable.id] ?: continue
			val restDelta = cellDelta(drawable, emptyMap(), puppet.parameters.associate { it.id.raw to it.default })
			// local -> world is affine while the angle axes are collapsed: fit it once per drawable
			val local = FloatArray(n * 2) { mesh.positions[it] + (restDelta?.get(it) ?: 0f) }
			val affine = fitAffine(local, world)
			println("  ${drawable.id.raw}: affine residual %.3f px".format(affine.residual))
			for ((cellParams, delta) in cells(drawable, puppet.parameters.associate { it.id.raw to it.default })) {
				val cellLocal = FloatArray(n * 2) { mesh.positions[it] + delta[it] }
				val cellWorld = affine.apply(cellLocal)
				sets += RigKeyformSetEdit(
					RigTargetRef(RigTargetKind.ART_MESH, drawable.id.raw),
					cellParams + mapOf("ParamAngleX" to 0f, "ParamAngleY" to 0f),
					RigKeyformGeometryEdit(positionDeltas = delta.toList()),
				)
				for (field in fields) {
					val out = FloatArray(n * 2)
					for (v in 0 until n) {
						// world space is y-up (world y = -canvas y); the field is in canvas pixels
						val (cx, cy) = field.sample(cellWorld[v * 2], -cellWorld[v * 2 + 1])
						val (lx, ly) = affine.inverseLinear(cx, -cy)
						out[v * 2] = delta[v * 2] + lx
						out[v * 2 + 1] = delta[v * 2 + 1] + ly
					}
					sets += RigKeyformSetEdit(
						RigTargetRef(RigTargetKind.ART_MESH, drawable.id.raw),
						cellParams + mapOf("ParamAngleX" to field.ax, "ParamAngleY" to field.ay),
						RigKeyformGeometryEdit(positionDeltas = out.toList()),
					)
				}
			}
		}
		println("retarget: ${neutralized.size} deformer keyforms neutralized, ${sets.size} drawable keyforms")
		return RigEditOverlay(keyformSetEdits = neutralized + sets)
	}

	/** Every existing keyform cell of [drawable] as (parameter coordinate, position deltas). */
	private fun cells(drawable: Drawable, defaults: Map<String, Float>): List<Pair<Map<String, Float>, FloatArray>> {
		val grid = drawable.geometryGrid
		val size = drawable.mesh!!.positions.size
		if (grid == null || grid.axes.isEmpty()) {
			val d = grid?.cells?.firstOrNull()?.form?.positionDeltas ?: FloatArray(size)
			return listOf(emptyMap<String, Float>() to d)
		}
		return grid.cells.map { cell ->
			val coord = grid.axes.mapIndexed { i, axis -> axis.parameterId.raw to axis.keys[cell.coordinate[i]] }.toMap()
			coord to cell.form.positionDeltas
		}
	}

	private fun cellDelta(drawable: Drawable, at: Map<String, Float>, defaults: Map<String, Float>): FloatArray? {
		val grid = drawable.geometryGrid ?: return null
		val want = grid.axes.map { axis -> axis.keys.indexOfFirst { abs(it - (at[axis.parameterId.raw] ?: defaults.getValue(axis.parameterId.raw))) < 1e-4 } }
		return grid.cells.firstOrNull { it.coordinate.toList() == want }?.form?.positionDeltas
	}

	private class Affine(val a: Double, val b: Double, val c: Double, val d: Double, val e: Double, val f: Double, val residual: Double) {
		// world = (a*x + b*y + c, d*x + e*y + f)
		fun apply(p: FloatArray) = FloatArray(p.size) { i ->
			val x = p[i - i % 2].toDouble()
			val y = p[i - i % 2 + 1].toDouble()
			(if (i % 2 == 0) a * x + b * y + c else d * x + e * y + f).toFloat()
		}
		fun inverseLinear(wx: Float, wy: Float): Pair<Float, Float> {
			val det = a * e - b * d
			return ((e * wx - b * wy) / det).toFloat() to ((-d * wx + a * wy) / det).toFloat()
		}
	}

	private fun fitAffine(local: FloatArray, world: FloatArray): Affine {
		val n = local.size / 2
		// normal equations for [x y 1] -> wx and -> wy
		val m = Array(3) { DoubleArray(3) }
		val rx = DoubleArray(3)
		val ry = DoubleArray(3)
		for (v in 0 until n) {
			val row = doubleArrayOf(local[v * 2].toDouble(), local[v * 2 + 1].toDouble(), 1.0)
			for (i in 0..2) {
				for (j in 0..2) m[i][j] += row[i] * row[j]
				rx[i] += row[i] * world[v * 2]
				ry[i] += row[i] * world[v * 2 + 1]
			}
		}
		val sx = solve3(m, rx)
		val sy = solve3(m, ry)
		var err = 0.0
		for (v in 0 until n) {
			val x = local[v * 2].toDouble()
			val y = local[v * 2 + 1].toDouble()
			err = maxOf(err, abs(sx[0] * x + sx[1] * y + sx[2] - world[v * 2]), abs(sy[0] * x + sy[1] * y + sy[2] - world[v * 2 + 1]))
		}
		return Affine(sx[0], sx[1], sx[2], sy[0], sy[1], sy[2], err)
	}

	private fun solve3(m0: Array<DoubleArray>, r0: DoubleArray): DoubleArray {
		val m = Array(3) { i -> m0[i].copyOf() + r0[i] }
		for (col in 0..2) {
			val p = (col..2).maxBy { abs(m[it][col]) }
			val t = m[col]; m[col] = m[p]; m[p] = t
			for (r in 0..2) if (r != col) {
				val k = m[r][col] / m[col][col]
				for (c in col..3) m[r][c] -= k * m[col][c]
			}
		}
		return DoubleArray(3) { m[it][3] / m[it][it] }
	}
}
