package org.astra.collector.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.astra.collector.data.model.CollectionTask
import org.astra.collector.data.model.CollectorState
import org.astra.collector.ui.theme.*

@Composable
fun UploadStatusScreen(
    task: CollectionTask,
    state: CollectorState,
    progressPercent: Int,
    errorMessage: String?,
    onRetry: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(SpaceBackground)
            .padding(24.dp),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier.fillMaxWidth().wrapContentHeight(),
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(16.dp),
            elevation = CardDefaults.cardElevation(8.dp)
        ) {
            Column(
                modifier = Modifier.padding(28.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Icon(
                    imageVector = Icons.Default.CloudUpload,
                    contentDescription = null,
                    tint = if (state == CollectorState.UPLOAD_FAILED) RedCritical else CyanAccent,
                    modifier = Modifier.size(54.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = when (state) {
                        CollectorState.VERIFYING -> "VERIFYING INTEGRITY..."
                        CollectorState.UPLOAD_FAILED -> "UPLOAD FAILED"
                        else -> "UPLOADING EXPERIMENT VIDEO..."
                    },
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = if (state == CollectorState.UPLOAD_FAILED) RedCritical else CyanAccent,
                    fontFamily = FontFamily.Monospace
                )

                Text(
                    text = "${task.experimentId} • ${task.runId} • ${task.cameraId}",
                    fontSize = 13.sp,
                    color = TextSecondary,
                    modifier = Modifier.padding(top = 4.dp, bottom = 24.dp)
                )

                if (state == CollectorState.UPLOAD_FAILED) {
                    Text(
                        text = errorMessage ?: "Network interruption encountered. Local recording remains safely stored on phone.",
                        color = RedCritical,
                        fontSize = 14.sp,
                        modifier = Modifier.padding(bottom = 24.dp)
                    )

                    Button(
                        onClick = onRetry,
                        modifier = Modifier.fillMaxWidth().height(50.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = CyanAccent),
                        shape = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.Refresh, contentDescription = null, tint = SpaceBackground)
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("RETRY UPLOAD", color = SpaceBackground, fontWeight = FontWeight.Bold)
                    }
                } else {
                    LinearProgressIndicator(
                        progress = { progressPercent / 100f },
                        modifier = Modifier.fillMaxWidth().height(10.dp),
                        color = CyanAccent,
                        trackColor = SurfaceCard
                    )

                    Spacer(modifier = Modifier.height(14.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = if (state == CollectorState.VERIFYING) "Verifying Hugging Face commit..." else "Streaming 8 MB chunks",
                            fontSize = 13.sp,
                            color = TextSecondary
                        )
                        Text(
                            text = "$progressPercent%",
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            color = CyanAccent,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
        }
    }
}
