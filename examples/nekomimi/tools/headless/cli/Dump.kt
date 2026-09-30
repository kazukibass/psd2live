package io.github.psd2live.ui

import io.github.psd2live.core.RigPreviewModel
import org.umamo.runtime.model.Deformer

/** Prints the deformer tree and keyform axes of a built rig (debug aid for keyform authoring). */
object RigDump {
	fun print(model: RigPreviewModel) {
		val p = model.rig.puppet
		println("canvas=${p.canvasWidth}x${p.canvasHeight} origin=${p.worldOriginX},${p.worldOriginY} ppu=${p.pixelsPerUnit}")
		for (d in p.deformers) {
			val grid = when (d) { is Deformer.Warp -> d.geometryGrid; is Deformer.Rotation -> d.geometryGrid; else -> null }
			val axes = grid?.axes?.joinToString { "${it.parameterId.raw}${it.keys.toList()}" }
			val extra = if (d is Deformer.Warp) " warp ${d.rows}x${d.columns}" else if (d is Deformer.Rotation) " rot" else ""
			println("DEF ${d.id.raw} '${d.name}' parent=${d.parent?.raw}$extra axes=[$axes]")
		}
		for (dr in p.drawables) {
			println("DRW ${dr.id.raw} '${dr.name}' parent=${dr.parentDeformerId?.raw} verts=${dr.mesh?.let { it.positions.size / 2 }} axes=[${dr.geometryGrid?.axes?.joinToString { "${it.parameterId.raw}${it.keys.toList()}" }}]")
		}
	}
}
