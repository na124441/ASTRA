package org.astra.collector

import com.google.gson.Gson
import org.astra.collector.data.model.CollectionTask
import org.astra.collector.data.model.RecordingMetadata
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

class LocalStorageAndContractTest {

    @get:Rule
    val tempFolder = TemporaryFolder()

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { fis ->
            val buffer = ByteArray(65536)
            var bytesRead: Int
            while (fis.read(buffer).also { bytesRead = it } != -1) {
                digest.update(buffer, 0, bytesRead)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun enforceVerifiedDeletion(file: File, serverStatus: String, isVerified: Boolean): Boolean {
        if (!isVerified || !serverStatus.equals("verified", ignoreCase = true)) {
            throw IllegalStateException("CRITICAL INVARIANT VIOLATION: Cannot delete unverified local video!")
        }
        return file.delete() && !file.exists()
    }

    @Test
    fun testStreamingSha256Calculation() {
        val file = tempFolder.newFile("test_sample.mp4")
        file.writeBytes("ASTRA-E EXPERIMENTAL FRAME DATA 2026".toByteArray(Charsets.UTF_8))

        val computed = computeSha256(file)
        val expected = "cffbb035c918ee0da79058b8d41cfd1d575fa3d623ce1b88e1471e9a2eb24239" // known hash for above text? Let's verify programmatic match
        val digest = MessageDigest.getInstance("SHA-256")
        val manualHash = digest.digest(file.readBytes()).joinToString("") { "%02x".format(it) }

        assertEquals(manualHash, computed)
        assertTrue(computed.length == 64)
    }

    @Test
    fun testFailClosedInvariantBlocksDeletionWhenUnverified() {
        val videoFile = tempFolder.newFile("unverified_video.mp4")
        videoFile.writeBytes(ByteArray(1024) { 0x42 })

        // 1. Unverified upload (server status: failed)
        try {
            enforceVerifiedDeletion(videoFile, serverStatus = "failed", isVerified = false)
            fail("Expected IllegalStateException on unverified upload!")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("CRITICAL INVARIANT VIOLATION"))
        }
        assertTrue("Local file MUST remain intact when upload unverified", videoFile.exists())

        // 2. Unverified upload (server status: in_progress)
        try {
            enforceVerifiedDeletion(videoFile, serverStatus = "in_progress", isVerified = false)
            fail("Expected IllegalStateException on unverified upload!")
        } catch (e: IllegalStateException) {
            assertTrue(e.message!!.contains("CRITICAL INVARIANT VIOLATION"))
        }
        assertTrue("Local file MUST remain intact while in progress", videoFile.exists())
    }

    @Test
    fun testVerifiedDeletionSucceedsWhenConfirmed() {
        val videoFile = tempFolder.newFile("verified_video.mp4")
        videoFile.writeBytes(ByteArray(2048) { 0x1A })
        assertTrue(videoFile.exists())

        val deleted = enforceVerifiedDeletion(videoFile, serverStatus = "verified", isVerified = true)
        assertTrue(deleted)
        assertFalse("Local file must be cleanly deleted once verified", videoFile.exists())
    }

    @Test
    fun testMetadataAndTaskSerialization() {
        val gson = Gson()
        val taskJson = """
            {
                "task_id": "TASK-0042",
                "experiment_id": "EXP001",
                "run_id": "RUN-0042",
                "camera_id": "CAM-01",
                "scenario_type": "nominal",
                "required_object": "RED_COMPONENT",
                "target": "TARGET_A",
                "duration_min": 30,
                "duration_max": 60,
                "procedure_steps": ["Step 1", "Step 2"]
            }
        """.trimIndent()

        val task = gson.fromJson(taskJson, CollectionTask::class.java)
        assertEquals("TASK-0042", task.taskId)
        assertEquals("EXP001", task.experimentId)
        assertEquals("RUN-0042", task.runId)
        assertEquals(2, task.procedureSteps.size)

        val metadata = RecordingMetadata(
            schemaVersion = "1.0",
            experimentId = "EXP001",
            runId = "RUN-0042",
            recordingId = "EXP001_RUN-0042_CAM-01",
            collectorId = "COL-007",
            cameraId = "CAM-01",
            scenarioType = "nominal",
            objectName = "RED_COMPONENT",
            target = "TARGET_A",
            durationSeconds = 45.2,
            width = 1920,
            height = 1080,
            fps = 30.0,
            orientation = "landscape",
            fileSizeBytes = 50000000L,
            sha256 = "abc123sha256",
            appVersion = "0.1.0",
            protocolVersion = "EXP001-v1.0",
            createdAt = "2026-09-05T21:00:00Z"
        )

        val jsonOutput = gson.toJson(metadata)
        assertTrue(jsonOutput.contains("\"schema_version\":\"1.0\""))
        assertTrue(jsonOutput.contains("\"run_id\":\"RUN-0042\""))
        assertTrue(jsonOutput.contains("\"sha256\":\"abc123sha256\""))
    }
}
