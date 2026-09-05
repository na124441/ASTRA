package org.astra.collector.camera

import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * High-reliability CameraX recording controller configured for standardized 1080p capture.
 */
class RecordingManager(private val context: Context) {

    private val tag = "AstraCamera"

    private var videoCapture: VideoCapture<Recorder>? = null
    private var activeRecording: Recording? = null
    private var timerJob: Job? = null

    private val _recordingDurationSeconds = MutableStateFlow(0)
    val recordingDurationSeconds: StateFlow<Int> = _recordingDurationSeconds.asStateFlow()

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    /**
     * Build and bind CameraX VideoCapture with Quality.FHD (1080p) fallback to HD.
     */
    fun bindCamera(
        lifecycleOwner: LifecycleOwner,
        cameraSelector: CameraSelector = CameraSelector.DEFAULT_BACK_CAMERA,
        onBound: (VideoCapture<Recorder>) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                val qualitySelector = QualitySelector.fromOrderedList(
                    listOf(Quality.FHD, Quality.HD, Quality.SD)
                )
                val recorder = Recorder.Builder()
                    .setQualitySelector(qualitySelector)
                    .build()

                videoCapture = VideoCapture.withOutput(recorder)
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, videoCapture)
                videoCapture?.let(onBound)
                Log.i(tag, "CameraX bound successfully for 1080p video recording")
            } catch (e: Exception) {
                Log.e(tag, "Failed to bind CameraX provider: ${e.message}", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    /**
     * Start video recording to destination file with HUD duration timer.
     */
    @SuppressLint("MissingPermission")
    fun startRecording(
        outputFile: File,
        scope: CoroutineScope,
        onFinalized: (File, Long) -> Unit,
        onError: (String) -> Unit
    ) {
        val capture = videoCapture ?: run {
            onError("Camera capture not initialized")
            return
        }

        val outputOptions = FileOutputOptions.Builder(outputFile).build()

        _recordingDurationSeconds.value = 0
        _isRecording.value = true

        activeRecording = capture.output
            .prepareRecording(context, outputOptions)
            .withAudioEnabled()
            .start(ContextCompat.getMainExecutor(context)) { event ->
                when (event) {
                    is VideoRecordEvent.Start -> {
                        Log.i(tag, "Video recording started: ${outputFile.name}")
                        startTimer(scope)
                    }
                    is VideoRecordEvent.Finalize -> {
                        stopTimer()
                        _isRecording.value = false
                        if (!event.hasError()) {
                            Log.i(tag, "Video recording finalized: ${outputFile.name} (${outputFile.length()} bytes)")
                            onFinalized(outputFile, outputFile.length())
                        } else {
                            val msg = "Recording finalize error: code ${event.error}"
                            Log.e(tag, msg)
                            onError(msg)
                        }
                    }
                }
            }
    }

    /**
     * Stop active recording and flush media muxer.
     */
    fun stopRecording() {
        activeRecording?.stop()
        activeRecording = null
        stopTimer()
        _isRecording.value = false
    }

    private fun startTimer(scope: CoroutineScope) {
        timerJob?.cancel()
        timerJob = scope.launch(Dispatchers.Default) {
            while (true) {
                delay(1000)
                _recordingDurationSeconds.value += 1
            }
        }
    }

    private fun stopTimer() {
        timerJob?.cancel()
        timerJob = null
    }
}
