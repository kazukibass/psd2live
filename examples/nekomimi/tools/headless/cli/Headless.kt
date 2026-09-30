import io.github.psd2live.core.PSD2LivePipeline
import io.github.psd2live.core.PipelineConfig
import io.github.psd2live.core.ProgressListener
import java.nio.file.Path

// Headless CLI: same options as Main.kt's CLI path, without the Compose GUI.
fun main(args: Array<String>) {
	val m = args.toList().windowed(2, 1).filter { it[0].startsWith("--") }.associate { it[0] to it[1] }
	val config = PipelineConfig(
		atlasSize = m["--atlas"]?.toInt() ?: 4096,
		meshSpacing = m["--mesh-spacing"]?.toInt() ?: 64,
		headTurnStrength = m["--head-strength"]?.toFloat() ?: 1f,
		bodyStrength = m["--body-strength"]?.toFloat() ?: 1f,
	)
	val input = Path.of(m.getValue("--input"))
	val output = Path.of(m.getValue("--output"))
	val progress = ProgressListener { stage, f -> println("%3d%%  %s".format((f * 100).toInt(), stage)) }
	var result = PSD2LivePipeline().run(input, output, config, progress)
	m["--retarget-vertices"]?.let { path ->
		// second pass: head turns from per-vertex displacements (tools/head3d.py)
		val overlay = io.github.psd2live.ui.HeadRetarget.buildFromVertices(result.previewModel, java.io.File(path))
		result = PSD2LivePipeline().run(input, output, config.copy(rigEdits = overlay), progress)
	}
	m["--retarget"]?.let { fields ->
		// second pass: replace the generated head-turn keyforms with the measured targets
		val overlay = io.github.psd2live.ui.HeadRetarget.build(result.previewModel, java.io.File(fields))
		result = PSD2LivePipeline().run(input, output, config.copy(rigEdits = overlay), progress)
	}
	result.exportedFiles.forEach { println("  ${it.path} (${it.bytes} bytes)") }
	result.warnings.forEach { System.err.println("WARN: $it") }
	if (args.contains("--dump")) io.github.psd2live.ui.RigDump.print(result.previewModel)
	m["--export-geometry"]?.let { path ->
		// rest + the 8 key extremes + half-way poses, mouth fully open (a closed mouth is a degenerate line)
		val poses = mutableListOf<Pair<String, Map<String, Float>>>()
		for (f in listOf(1f, 0.5f)) for (ay in listOf(30f, 0f, -30f)) for (ax in listOf(-45f, 0f, 45f)) {
			if (f < 1f && ax == 0f && ay == 0f) continue
			val name = if (ax == 0f && ay == 0f) "rest" else "x%+d_y%+d".format((ax * f).toInt(), (ay * f).toInt())
			poses += name to mapOf("ParamAngleX" to ax * f, "ParamAngleY" to ay * f, "ParamMouthOpenY" to 1f)
		}
		io.github.psd2live.ui.GeometryExport.write(result.previewModel, poses, java.io.File(path))
	}
	m["--render"]?.let { dir ->
		val model = result.previewModel
		println("PARAMS: " + model.rig.puppet.parameters.joinToString { "${it.id.raw}[${it.min},${it.max}]" })
		// the key extremes, i.e. the poses the reference sheet describes
		val x = 45f
		val y = 30f
		val nine = listOf(
			"up-left" to mapOf("ParamAngleX" to -x, "ParamAngleY" to y), "up" to mapOf("ParamAngleY" to y), "up-right" to mapOf("ParamAngleX" to x, "ParamAngleY" to y),
			"left" to mapOf("ParamAngleX" to -x), "front" to mapOf(), "right" to mapOf("ParamAngleX" to x),
			"down-left" to mapOf("ParamAngleX" to -x, "ParamAngleY" to -y), "down" to mapOf("ParamAngleY" to -y), "down-right" to mapOf("ParamAngleX" to x, "ParamAngleY" to -y),
		)
		val face = listOf(
			"mouth closed" to mapOf("ParamMouthOpenY" to 0f), "mouth half" to mapOf("ParamMouthOpenY" to 0.5f), "mouth open" to mapOf("ParamMouthOpenY" to 1f),
			"eyes closed" to mapOf("ParamEyeLOpen" to 0f, "ParamEyeROpen" to 0f, "ParamMouthOpenY" to 0f), "look left" to mapOf("ParamEyeBallX" to -1f, "ParamMouthOpenY" to 0f), "tilt Z" to mapOf("ParamAngleZ" to 20f, "ParamMouthOpenY" to 0f),
			"body X" to mapOf("ParamBodyAngleX" to 10f, "ParamMouthOpenY" to 0f), "body Y" to mapOf("ParamBodyAngleY" to 10f, "ParamMouthOpenY" to 0f), "breath" to mapOf("ParamBreath" to 1f, "ParamMouthOpenY" to 0f),
		)
		val closedMouth = nine.map { (k, v) -> k to (v + ("ParamMouthOpenY" to 0f)) }
		io.github.psd2live.ui.PoseSheet.render(model, closedMouth, intArrayOf(225, -30, 1350, 900), 512, java.io.File(dir, "poses_head.png"))
		io.github.psd2live.ui.PoseSheet.render(model, face, intArrayOf(650, 350, 500, 450), 512, java.io.File(dir, "poses_face.png"))
		io.github.psd2live.ui.PoseSheet.render(model, listOf("front" to mapOf("ParamMouthOpenY" to 0f), "body X" to mapOf("ParamBodyAngleX" to 10f, "ParamMouthOpenY" to 0f), "turn" to mapOf("ParamAngleX" to 30f, "ParamBodyAngleX" to 10f, "ParamMouthOpenY" to 0f)), intArrayOf(100, 0, 1600, 1600), 512, java.io.File(dir, "poses_body.png"))
	}
}
