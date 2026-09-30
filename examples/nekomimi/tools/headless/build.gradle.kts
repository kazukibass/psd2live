plugins {
	kotlin("jvm") version "2.4.10"
	kotlin("plugin.serialization") version "2.4.10"
	application
}
kotlin { jvmToolchain(21) }
// psd2live sources, relative to this directory (examples/nekomimi/tools/headless)
val src = rootDir.resolve("../../../../src/main")
sourceSets.main {
	kotlin.setSrcDirs(listOf(src.resolve("kotlin"), file("cli")))
	kotlin.exclude({ it.path.startsWith("io/github/psd2live/ui/") && it.path != "io/github/psd2live/ui/RigCanvasSupport.kt" })
	kotlin.exclude( "io/github/psd2live/agent/**", "io/github/psd2live/history/**", "io/github/psd2live/project/**", "io/github/psd2live/Main.kt")
	resources.setSrcDirs(listOf(src.resolve("resources")))
}
dependencies {
	implementation(platform("io.ktor:ktor-bom:3.5.1"))
	implementation(kotlin("reflect"))
	implementation("org.jdom:jdom:1.1.3")
	implementation("com.squareup.okio:okio:3.17.0")
	implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.8.0")
	implementation("app.cash.sqldelight:sqlite-driver:2.0.2")
	implementation(platform("org.lwjgl:lwjgl-bom:3.4.2"))
	implementation("org.lwjgl:lwjgl")
	implementation("org.lwjgl:lwjgl-opengl")
	runtimeOnly("org.lwjgl:lwjgl::natives-linux")
	runtimeOnly("org.lwjgl:lwjgl-opengl::natives-linux")
	implementation("net.java.dev.jna:jna:5.18.0")
	implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
	implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
}
application { mainClass.set("HeadlessKt"); applicationDefaultJvmArgs = listOf("-Xmx6g", "-Djava.awt.headless=true") }
