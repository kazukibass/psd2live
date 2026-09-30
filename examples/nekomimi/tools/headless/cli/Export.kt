package io.github.psd2live.ui

import io.github.psd2live.core.RigPreviewModel
import org.umamo.runtime.model.Deformer
import org.umamo.runtime.model.DeformerId
import org.umamo.runtime.model.ParameterId
import java.io.File

/**
 * Writes every drawable's rest geometry and its world positions at a set of poses as JSON, in canvas
 * pixels (y down), for the Python analysis / harness (tools/harness.py).
 */
object GeometryExport {
	fun write(model: RigPreviewModel, poses: List<Pair<String, Map<String, Float>>>, out: File) {
		val puppet = model.rig.puppet
		val byId = puppet.deformers.associateBy { it.id }
		fun chain(id: DeformerId?): List<String> {
			val names = mutableListOf<String>()
			var cur = id
			while (cur != null) { names += cur.raw; cur = byId[cur]?.parent }
			return names
		}
		val evaluated = poses.map { (name, params) -> name to RigCanvasSupport.evaluate(model, params.mapKeys { ParameterId(it.key) }) }
		out.bufferedWriter().use { w ->
			w.write("{\"poses\":[")
			w.write(poses.joinToString(",") { (name, params) ->
				"{\"name\":\"$name\",\"params\":{" + params.entries.joinToString(",") { "\"${it.key}\":${it.value}" } + "}}"
			})
			w.write("],\"drawables\":[")
			var first = true
			for (d in puppet.drawables) {
				val mesh = d.mesh ?: continue
				if (!first) w.write(",")
				first = false
				w.write("{\"id\":\"${d.id.raw}\",\"name\":\"${d.name.replace("\"", "'")}\",\"chain\":[")
				w.write(chain(d.parentDeformerId).joinToString(",") { "\"$it\"" })
				w.write("],\"indices\":[")
				w.write(mesh.indices.joinToString(","))
				w.write("],\"world\":{")
				w.write(evaluated.joinToString(",") { (name, geo) ->
					val p = geo.worldPositions[d.id]
					"\"$name\":" + if (p == null) "null" else
						"[" + p.indices.joinToString(",") { i -> "%.2f".format(if (i % 2 == 0) p[i] else -p[i]) } + "]"
				})
				w.write("}}")
			}
			w.write("]}")
		}
	}
}
