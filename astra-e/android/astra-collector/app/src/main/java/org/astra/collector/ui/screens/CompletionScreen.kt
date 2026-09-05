package org.astra.collector.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.DeleteSweep
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
import org.astra.collector.ui.theme.*

@Composable
fun CompletionScreen(
    task: CollectionTask,
    remotePath: String,
    onNextTask: () -> Unit
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
                    imageVector = Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = GreenSuccess,
                    modifier = Modifier.size(64.dp)
                )

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "DATASET UPLOAD VERIFIED",
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = GreenSuccess,
                    fontFamily = FontFamily.Monospace
                )

                Text(
                    text = "Task ${task.taskId} • ${task.runId} (${task.cameraId}) Complete",
                    fontSize = 14.sp,
                    color = TextSecondary,
                    modifier = Modifier.padding(top = 4.dp, bottom = 20.dp)
                )

                // Remote persistence pill
                Surface(
                    color = SurfaceCard,
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Text("REMOTE HUGGING FACE TARGET", fontSize = 11.sp, color = TextSecondary, fontWeight = FontWeight.SemiBold)
                        Text(
                            text = remotePath,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium,
                            color = CyanAccent,
                            fontFamily = FontFamily.Monospace,
                            modifier = Modifier.padding(top = 4.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                // Local cleanup confirmed pill
                Surface(
                    color = SurfaceCard,
                    shape = RoundedCornerShape(8.dp),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Row(
                        modifier = Modifier.padding(14.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.DeleteSweep, contentDescription = null, tint = GreenSuccess, modifier = Modifier.size(20.dp))
                        Spacer(modifier = Modifier.width(10.dp))
                        Column {
                            Text("LOCAL TEMPORARY STORAGE", fontSize = 11.sp, color = TextSecondary, fontWeight = FontWeight.SemiBold)
                            Text("Verified deleted & storage reclaimed", fontSize = 13.sp, color = TextPrimary, fontWeight = FontWeight.Medium)
                        }
                    }
                }

                Spacer(modifier = Modifier.height(28.dp))

                Button(
                    onClick = onNextTask,
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = CyanAccent),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text(
                        text = "DISPENSE NEXT TASK",
                        color = SpaceBackground,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Icon(Icons.Default.ArrowForward, contentDescription = null, tint = SpaceBackground)
                }
            }
        }
    }
}
