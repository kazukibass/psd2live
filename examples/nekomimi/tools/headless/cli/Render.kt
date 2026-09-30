package io.github.psd2live.ui

import io.github.psd2live.core.RigPreviewModel
import org.umamo.runtime.model.ParameterId
import java.awt.Color
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/** Renders labelled poses of a built rig into one contact sheet (software rasteriser, no GPU). */
object PoseSheet {
	fun render(model: RigPreviewModel, poses: List<Pair<String, Map<String, Float>>>, crop: IntArray, cellW: Int, out: File) {
		val (cx, cy, cw, ch) = crop.toList()
		val scale = cellW.toDouble() / cw
		val cellH = (ch * scale).toInt()
		val cols = 3
		val rows = (poses.size + cols - 1) / cols
		val sheet = BufferedImage(cols * cellW, rows * cellH, BufferedImage.TYPE_INT_RGB)
		val g = sheet.createGraphics()
		g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
		val cw0 = model.analysis.source.widthPx.toFloat()
		val ch0 = model.analysis.source.heightPx.toFloat()
		poses.forEachIndexed { i, (label, params) ->
			val ox = (i % cols) * cellW
			val oy = (i / cols) * cellH
			val cell = g.create(ox, oy, cellW, cellH) as java.awt.Graphics2D
			cell.color = Color(250, 250, 250)
			cell.fillRect(0, 0, cellW, cellH)
			val viewport = CanvasViewport(scale, -cx * scale, -cy * scale, cw0, ch0)
			val geometry = RigCanvasSupport.evaluate(model, params.mapKeys { ParameterId(it.key) })
			RigCanvasSupport.paintTexturedRig(cell, model, geometry, viewport)
			cell.color = Color(40, 40, 40)
			cell.fillRect(6, 6, label.length * 8 + 12, 20)
			cell.color = Color.WHITE
			cell.drawString(label, 12, 21)
			cell.dispose()
		}
		g.dispose()
		ImageIO.write(sheet, "png", out)
	}
}
