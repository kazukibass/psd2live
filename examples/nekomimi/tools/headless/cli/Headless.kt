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
	val result = PSD2LivePipeline().run(input, output, config, ProgressListener { stage, f -> println("%3d%%  %s".format((f * 100).toInt(), stage)) })
	result.exportedFiles.forEach { println("  ${it.path} (${it.bytes} bytes)") }
	result.warnings.forEach { System.err.println("WARN: $it") }
	m["--render"]?.let { dir ->
		val model = result.previewModel
		println("PARAMS: " + model.rig.puppet.parameters.joinToString { "${it.id.raw}[${it.min},${it.max}]" })
		val a = 30f
		val nine = listOf(
			"up-left" to mapOf("ParamAngleX" to -a, "ParamAngleY" to a), "up" to mapOf("ParamAngleY" to a), "up-right" to mapOf("ParamAngleX" to a, "ParamAngleY" to a),
			"left" to mapOf("ParamAngleX" to -a), "front" to mapOf(), "right" to mapOf("ParamAngleX" to a),
			"down-left" to mapOf("ParamAngleX" to -a, "ParamAngleY" to -a), "down" to mapOf("ParamAngleY" to -a), "down-right" to mapOf("ParamAngleX" to a, "ParamAngleY" to -a),
		)
		val face = listOf(
			"mouth closed" to mapOf("ParamMouthOpenY" to 0f), "mouth half" to mapOf("ParamMouthOpenY" to 0.5f), "mouth open" to mapOf("ParamMouthOpenY" to 1f),
			"eyes closed" to mapOf("ParamEyeLOpen" to 0f, "ParamEyeROpen" to 0f, "ParamMouthOpenY" to 0f), "look left" to mapOf("ParamEyeBallX" to -1f, "ParamMouthOpenY" to 0f), "tilt Z" to mapOf("ParamAngleZ" to 20f, "ParamMouthOpenY" to 0f),
			"body X" to mapOf("ParamBodyAngleX" to 10f, "ParamMouthOpenY" to 0f), "body Y" to mapOf("ParamBodyAngleY" to 10f, "ParamMouthOpenY" to 0f), "breath" to mapOf("ParamBreath" to 1f, "ParamMouthOpenY" to 0f),
		)
		val closedMouth = nine.map { (k, v) -> k to (v + ("ParamMouthOpenY" to 0f)) }
		io.github.psd2live.ui.PoseSheet.render(model, closedMouth, intArrayOf(450, 0, 900, 900), 512, java.io.File(dir, "poses_head.png"))
		io.github.psd2live.ui.PoseSheet.render(model, face, intArrayOf(650, 350, 500, 450), 512, java.io.File(dir, "poses_face.png"))
		io.github.psd2live.ui.PoseSheet.render(model, listOf("front" to mapOf("ParamMouthOpenY" to 0f), "body X" to mapOf("ParamBodyAngleX" to 10f, "ParamMouthOpenY" to 0f), "turn" to mapOf("ParamAngleX" to 30f, "ParamBodyAngleX" to 10f, "ParamMouthOpenY" to 0f)), intArrayOf(100, 0, 1600, 1600), 512, java.io.File(dir, "poses_body.png"))
	}
}
