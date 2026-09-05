package org.astra.collector.storage

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest

/**
 * Manages private temporary video storage, cryptographic SHA-256 generation,
 * and enforces the fail-closed local deletion invariant:
 * NO VERIFIED REMOTE UPLOAD -> NO LOCAL DELETE
 */
class LocalStorageManager(private val context: Context) {

    private val tag = "AstraLocalStorage"

    private val tempDir: File by lazy {
        val dir = File(context.filesDir, "temp_recordings")
        if (!dir.exists()) {
            dir.mkdirs()
        }
        dir
    }

    /**
     * Create a new temporary MP4 file descriptor inside private app storage.
     */
    fun createTempRecordingFile(runId: String, cameraId: String): File {
        val filename = "EXP001_${runId}_${cameraId}_${System.currentTimeMillis()}.mp4"
        return File(tempDir, filename)
    }

    /**
     * Compute whole-file SHA-256 checksum using streaming 64 KB buffer.
     */
    fun computeSha256(file: File): String {
        require(file.exists() && file.isFile) { "File does not exist: ${file.absolutePath}" }
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

    /**
     * Enforce strict fail-closed deletion:
     * Deletes the local recording ONLY after remote verification is confirmed.
     * Throws IllegalStateException if deletion is attempted on an unverified upload.
     */
    fun deleteVerifiedFile(file: File, serverStatus: String, isVerified: Boolean): Boolean {
        if (!isVerified || !serverStatus.equals("verified", ignoreCase = true)) {
            val errorMsg = "CRITICAL INVARIANT VIOLATION: Attempted local delete when isVerified=$isVerified, status=$serverStatus"
            Log.e(tag, errorMsg)
            throw IllegalStateException(errorMsg)
        }

        if (!file.exists()) {
            Log.i(tag, "Local file already deleted: ${file.name}")
            return true
        }

        val success = file.delete()
        val existsAfter = file.exists()

        if (success && !existsAfter) {
            Log.i(tag, "Local video securely deleted after verified remote upload: ${file.name}")
            return true
        } else {
            Log.w(tag, "Failed to delete local video file: ${file.absolutePath}")
            return false
        }
    }

    /**
     * List all unverified or pending files currently retained on disk.
     */
    fun getRetainedVideoFiles(): List<File> {
        return tempDir.listFiles { f -> f.extension == "mp4" }?.toList() ?: emptyList()
    }
}
