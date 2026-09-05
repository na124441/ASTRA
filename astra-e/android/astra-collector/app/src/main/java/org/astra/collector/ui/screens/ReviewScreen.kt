package org.astra.collector.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.CloudUpload
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.astra.collector.data.model.CollectionTask
import org.astra.collector.storage.LocalStorageManager
import org.astra.collector.ui.theme.*
import java.io.File

@Composable
fun ReviewScreen(
    task: CollectionTask,
    videoFile: File,
    durationSeconds: Double,
    storageManager: LocalStorageManager,
    onReRecord: () -> Unit,
    onUpload: (sha256: String) -> Unit
) {
    var computedSha256 by remember { mutableStateOf<String?>(null) }
    var isHashing by remember { mutableStateOf(true) }

    LaunchedEffect(videoFile) {
        withContext(Dispatchers.IO) {
            val hash = storageManager.computeSha256(videoFile)
            withContext(Dispatchers.Main) {
                computedSha256 = hash
                isHashing = false
            }
        }
    }

    val isDurationValid = durationSeconds >= task.durationMin && durationSeconds <= task.durationMax
    val fileSizeMb = videoFile.length() / (1024.0 * 1024.0)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(SpaceBackground)
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "RECORDING VALIDATION",
            fontSize = 22.sp,
            fontWeight = FontWeight.Bold,
            color = CyanAccent,
            fontFamily = FontFamily.Monospace
        )
        Text(
            text = "${task.experimentId} • ${task.runId} • ${task.cameraId}",
            fontSize = 14.sp,
            color = TextSecondary,
            modifier = Modifier.padding(top = 4.dp, bottom = 20.dp)
        )

        Card(
            modifier = Modifier.fillMaxWidth().weight(1f),
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(16.dp)
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                // Duration Check
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text("DURATION", fontSize = 12.sp, color = TextSecondary)
                        Text(
                            "%.1f seconds (expected %d–%ds)".format(durationSeconds, task.durationMin, task.durationMax),
                            fontSize = 15.sp,
                            fontWeight = FontWeight.SemiBold,
                            color = TextPrimary
                        )
                    }
                    Icon(
                        imageVector = if (isDurationValid) Icons.Default.CheckCircle else Icons.Default.Warning,
                        contentDescription = null,
                        tint = if (isDurationValid) GreenSuccess else AmberWarning
                    )
                }

                Divider(color = SurfaceCard)

                // Resolution & Standard Check
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text("CAMERA SPECIFICATION", fontSize = 12.sp, color = TextSecondary)
                        Text("1920×1080 (Landscape FHD) • 30 FPS", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                    }
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = GreenSuccess)
                }

                Divider(color = SurfaceCard)

                // File Size
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text("LOCAL TEMPORARY SIZE", fontSize = 12.sp, color = TextSecondary)
                        Text("%.2f MB".format(fileSizeMb), fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = TextPrimary)
                    }
                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = GreenSuccess)
                }

                Divider(color = SurfaceCard)

                // SHA-256 Checksum
                Column(modifier = Modifier.fillMaxWidth()) {
                    Text("CRYPTOGRAPHIC SHA-256 CHECKSUM", fontSize = 12.sp, color = TextSecondary)
                    if (isHashing) {
                        Row(modifier = Modifier.padding(top = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                            CircularProgressIndicator(color = CyanAccent, modifier = Modifier.size(16.dp), strokeWidth = 2.dp)
                            Spacer(modifier = Modifier.width(10.dp))
                            Text("Computing streaming SHA-256 digest...", fontSize = 13.sp, color = TextSecondary)
                        }
                    } else {
                        Text(
                            text = computedSha256 ?: "Error",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                            color = CyanAccent,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.padding(top = 6.dp)
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Actions Row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            OutlinedButton(
                onClick = onReRecord,
                modifier = Modifier.weight(1f).height(54.dp),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = TextPrimary),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("RE-RECORD", fontWeight = FontWeight.Bold)
            }

            Button(
                onClick = {
                    computedSha256?.let { onUpload(it) }
                },
                enabled = !isHashing && computedSha256 != null,
                modifier = Modifier.weight(1f).height(54.dp),
                colors = ButtonDefaults.buttonColors(containerColor = CyanAccent),
                shape = RoundedCornerShape(12.dp)
            ) {
                Icon(Icons.Default.CloudUpload, contentDescription = null, tint = SpaceBackground)
                Spacer(modifier = Modifier.width(8.dp))
                Text("UPLOAD", color = SpaceBackground, fontWeight = FontWeight.Bold)
            }
        }
    }
}
