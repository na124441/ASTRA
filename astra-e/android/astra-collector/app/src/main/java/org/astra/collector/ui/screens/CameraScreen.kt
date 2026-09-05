package org.astra.collector.ui.screens

import androidx.activity.compose.BackHandler
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import org.astra.collector.camera.RecordingManager
import org.astra.collector.data.model.CollectionTask
import org.astra.collector.ui.theme.*
import java.io.File

@Composable
fun CameraScreen(
    task: CollectionTask,
    recordingManager: RecordingManager,
    outputFile: File,
    onRecordingFinished: (File, Long) -> Unit,
    onError: (String) -> Unit
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    val durationSeconds by recordingManager.recordingDurationSeconds.collectAsState()
    val isRecording by recordingManager.isRecording.collectAsState()

    // Block back button during active recording to prevent accidental data loss
    BackHandler(enabled = isRecording) {
        // Intercepted: User must press STOP
    }

    // Flashing red dot animation
    val infiniteTransition = rememberInfiniteTransition(label = "rec_blink")
    val alphaAnim by infiniteTransition.animateFloat(
        initialValue = 1.0f,
        targetValue = 0.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )

    Box(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        // CameraX Viewfinder Preview
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                    recordingManager.bindCamera(lifecycleOwner) { _ ->
                        recordingManager.startRecording(
                            outputFile = outputFile,
                            scope = scope,
                            onFinalized = onRecordingFinished,
                            onError = onError
                        )
                    }
                }
            },
            modifier = Modifier.fillMaxSize()
        )

        // Top HUD Overlay
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp)
                .align(Alignment.TopCenter),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Task Metadata Pill
            Surface(
                color = Color.Black.copy(alpha = 0.65f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.border(1.dp, SurfaceCard, RoundedCornerShape(8.dp))
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "${task.experimentId} • ${task.runId} • ${task.cameraId}",
                        color = CyanAccent,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }

            // REC Blinking Timer
            Surface(
                color = Color.Black.copy(alpha = 0.65f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.border(1.dp, RedCritical.copy(alpha = 0.5f), RoundedCornerShape(8.dp))
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(
                        modifier = Modifier
                            .size(12.dp)
                            .clip(CircleShape)
                            .background(RedCritical)
                            .alpha(if (isRecording) alphaAnim else 1f)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    val mins = durationSeconds / 60
                    val secs = durationSeconds % 60
                    Text(
                        text = "REC %02d:%02d".format(mins, secs),
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        // Bottom Stop Button Bar
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp)
                .align(Alignment.BottomCenter),
            contentAlignment = Alignment.Center
        ) {
            Button(
                onClick = {
                    recordingManager.stopRecording()
                },
                modifier = Modifier
                    .width(220.dp)
                    .height(64.dp),
                colors = ButtonDefaults.buttonColors(containerColor = RedCritical),
                shape = RoundedCornerShape(16.dp),
                elevation = ButtonDefaults.buttonElevation(8.dp)
            ) {
                Icon(Icons.Default.Stop, contentDescription = null, tint = Color.White, modifier = Modifier.size(28.dp))
                Spacer(modifier = Modifier.width(12.dp))
                Text(
                    text = "STOP RECORDING",
                    color = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize = 16.sp
                )
            }
        }
    }
}
