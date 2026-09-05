package org.astra.collector

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.work.Constraints
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkInfo
import androidx.work.WorkManager
import androidx.work.workDataOf
import com.google.gson.Gson
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.astra.collector.camera.RecordingManager
import org.astra.collector.data.model.*
import org.astra.collector.data.remote.CollectorApiService
import org.astra.collector.storage.LocalStorageManager
import org.astra.collector.ui.screens.*
import org.astra.collector.ui.theme.AstraCollectorTheme
import org.astra.collector.ui.theme.SpaceBackground
import org.astra.collector.worker.UploadWorker
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

class MainActivity : ComponentActivity() {

    private lateinit var storageManager: LocalStorageManager
    private lateinit var recordingManager: RecordingManager

    private var serverUrlState by mutableStateOf("http://10.0.2.2:8000")
    private var collectorIdState by mutableStateOf("COL-007")
    private var currentState by mutableStateOf(CollectorState.READY)
    private var activeTask by mutableStateOf<CollectionTask?>(null)
    private var tempFile by mutableStateOf<File?>(null)
    private var recordedDurationSec by mutableStateOf(0.0)
    private var uploadProgress by mutableStateOf(0)
    private var remotePathState by mutableStateOf<String?>(null)
    private var uploadErrorState by mutableStateOf<String?>(null)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraGranted = permissions[Manifest.permission.CAMERA] ?: false
        val audioGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
        if (!cameraGranted || !audioGranted) {
            Toast.makeText(this, "Camera & Audio permissions required for ASTRA Collector", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        storageManager = LocalStorageManager(this)
        recordingManager = RecordingManager(this)

        checkAndRequestPermissions()

        setContent {
            AstraCollectorTheme {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(SpaceBackground)
                ) {
                    when {
                        activeTask == null -> {
                            ConnectScreen(
                                onConnected = { url, id ->
                                    serverUrlState = url
                                    collectorIdState = id
                                    fetchNextTask(url, id)
                                }
                            )
                        }
                        currentState == CollectorState.ASSIGNED -> {
                            TaskScreen(
                                task = activeTask!!,
                                onStartRecording = {
                                    val file = storageManager.createTempRecordingFile(
                                        runId = activeTask!!.runId,
                                        cameraId = activeTask!!.cameraId
                                    )
                                    tempFile = file
                                    currentState = CollectorState.RECORDING
                                }
                            )
                        }
                        currentState == CollectorState.RECORDING -> {
                            tempFile?.let { file ->
                                CameraScreen(
                                    task = activeTask!!,
                                    recordingManager = recordingManager,
                                    outputFile = file,
                                    onRecordingFinished = { finishedFile, sizeBytes ->
                                        recordedDurationSec = recordingManager.recordingDurationSeconds.value.toDouble()
                                        currentState = CollectorState.RECORDED
                                    },
                                    onError = { err ->
                                        Toast.makeText(this@MainActivity, "Recording Error: $err", Toast.LENGTH_LONG).show()
                                        currentState = CollectorState.ASSIGNED
                                    }
                                )
                            }
                        }
                        currentState == CollectorState.RECORDED -> {
                            tempFile?.let { file ->
                                ReviewScreen(
                                    task = activeTask!!,
                                    videoFile = file,
                                    durationSeconds = recordedDurationSec,
                                    storageManager = storageManager,
                                    onReRecord = {
                                        file.delete()
                                        currentState = CollectorState.ASSIGNED
                                    },
                                    onUpload = { sha256 ->
                                        startBackgroundUpload(file, sha256)
                                    }
                                )
                            }
                        }
                        currentState in listOf(CollectorState.UPLOADING, CollectorState.VERIFYING, CollectorState.UPLOAD_FAILED) -> {
                            UploadStatusScreen(
                                task = activeTask!!,
                                state = currentState,
                                progressPercent = uploadProgress,
                                errorMessage = uploadErrorState,
                                onRetry = {
                                    tempFile?.let { file ->
                                        val hash = storageManager.computeSha256(file)
                                        startBackgroundUpload(file, hash)
                                    }
                                }
                            )
                        }
                        currentState == CollectorState.COMPLETED -> {
                            CompletionScreen(
                                task = activeTask!!,
                                remotePath = remotePathState ?: "videos/exp001/${activeTask!!.runId}/${activeTask!!.cameraId}.mp4",
                                onNextTask = {
                                    activeTask = null
                                    currentState = CollectorState.ASSIGNED
                                    fetchNextTask(serverUrlState, collectorIdState)
                                }
                            )
                        }
                    }
                }
            }
        }
    }

    private fun checkAndRequestPermissions() {
        val permissions = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO
        )
        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            permissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun fetchNextTask(url: String, collectorId: String) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val api = CollectorApiService.create(url)
                // Register device first
                val regResp = api.registerDevice(
                    DeviceRegisterRequest(
                        collector_id = collectorId,
                        device_id = android.os.Build.ID,
                        app_version = "0.1.0",
                        device_model = "${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}"
                    )
                )
                // Fetch next task
                val task = api.getNextTask(collectorId)
                withContext(Dispatchers.Main) {
                    if (task != null) {
                        activeTask = task
                        currentState = CollectorState.ASSIGNED
                    } else {
                        Toast.makeText(this@MainActivity, "No pending tasks available for $collectorId", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Connection Failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun startBackgroundUpload(videoFile: File, sha256: String) {
        currentState = CollectorState.UPLOADING
        uploadProgress = 0
        uploadErrorState = null

        val task = activeTask ?: return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val api = CollectorApiService.create(serverUrlState)
                val totalBytes = videoFile.length()
                val chunkSize = 8 * 1024 * 1024
                val totalChunks = ((totalBytes + chunkSize - 1) / chunkSize).toInt()

                val isoFormatter = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
                    timeZone = TimeZone.getTimeZone("UTC")
                }

                val metadata = RecordingMetadata(
                    schema_version = "1.0",
                    experiment_id = task.experimentId,
                    run_id = task.runId,
                    recording_id = "${task.experimentId}_${task.runId}_${task.cameraId}",
                    collector_id = collectorIdState,
                    camera_id = task.cameraId,
                    scenario_type = task.scenarioType,
                    objectName = task.requiredObject,
                    target = task.target,
                    durationSeconds = recordedDurationSec,
                    width = 1920,
                    height = 1080,
                    fps = 30.0,
                    orientation = "landscape",
                    fileSizeBytes = totalBytes,
                    sha256 = sha256,
                    app_version = "0.1.0",
                    protocol_version = "EXP001-v1.0",
                    createdAt = isoFormatter.format(Date())
                )

                val initResp = api.initiateUpload(
                    UploadInitiateRequest(
                        task_id = task.taskId,
                        collector_id = collectorIdState,
                        file_size_bytes = totalBytes,
                        total_chunks = totalChunks,
                        sha256 = sha256,
                        metadata = metadata
                    )
                )

                val workData = workDataOf(
                    UploadWorker.KEY_UPLOAD_ID to initResp.uploadId,
                    UploadWorker.KEY_TASK_ID to task.taskId,
                    UploadWorker.KEY_FILE_PATH to videoFile.absolutePath,
                    UploadWorker.KEY_SHA256 to sha256,
                    UploadWorker.KEY_METADATA_JSON to Gson().toJson(metadata),
                    UploadWorker.KEY_SERVER_URL to serverUrlState
                )

                val constraints = Constraints.Builder()
                    .setRequiredNetworkType(NetworkType.CONNECTED)
                    .build()

                val uploadWorkRequest = OneTimeWorkRequestBuilder<UploadWorker>()
                    .setConstraints(constraints)
                    .setInputData(workData)
                    .build()

                val workManager = WorkManager.getInstance(this@MainActivity)
                workManager.enqueue(uploadWorkRequest)

                withContext(Dispatchers.Main) {
                    workManager.getWorkInfoByIdLiveData(uploadWorkRequest.id).observe(this@MainActivity) { info ->
                        if (info != null) {
                            when (info.state) {
                                WorkInfo.State.RUNNING -> {
                                    val progress = info.progress.getInt(UploadWorker.KEY_PROGRESS, 0)
                                    uploadProgress = progress
                                    if (progress >= 95) {
                                        currentState = CollectorState.VERIFYING
                                    }
                                }
                                WorkInfo.State.SUCCEEDED -> {
                                    remotePathState = info.outputData.getString(UploadWorker.KEY_REMOTE_PATH)
                                    currentState = CollectorState.COMPLETED
                                }
                                WorkInfo.State.FAILED -> {
                                    uploadErrorState = "Upload failed. Local file safely retained."
                                    currentState = CollectorState.UPLOAD_FAILED
                                }
                                else -> Unit
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    uploadErrorState = e.message
                    currentState = CollectorState.UPLOAD_FAILED
                }
            }
        }
    }
}
