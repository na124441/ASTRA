package org.astra.collector.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Videocam
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
fun TaskScreen(
    task: CollectionTask,
    onStartRecording: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(SpaceBackground)
            .padding(20.dp)
    ) {
        // Top Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "${task.experimentId} • ${task.runId}",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = CyanAccent,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    text = "Perspective: ${task.cameraId} (Landscape 1080p)",
                    fontSize = 13.sp,
                    color = TextSecondary
                )
            }

            val badgeColor = if (task.scenarioType.equals("nominal", ignoreCase = true)) GreenSuccess else AmberWarning
            Surface(
                color = badgeColor.copy(alpha = 0.2f),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.border(1.dp, badgeColor, RoundedCornerShape(8.dp))
            ) {
                Text(
                    text = task.scenarioType.uppercase(),
                    color = badgeColor,
                    fontWeight = FontWeight.Bold,
                    fontSize = 12.sp,
                    modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp)
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Target & Object Card
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(16.dp),
                horizontalArrangement = Arrangement.SpaceAround
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("REQUIRED OBJECT", fontSize = 11.sp, color = TextSecondary, fontWeight = FontWeight.SemiBold)
                    Text(task.requiredObject, fontSize = 16.sp, color = TextPrimary, fontWeight = FontWeight.Bold)
                }
                Divider(modifier = Modifier.height(36.dp).width(1.dp), color = SurfaceCard)
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("TARGET LOCATION", fontSize = 11.sp, color = TextSecondary, fontWeight = FontWeight.SemiBold)
                    Text(task.target, fontSize = 16.sp, color = TextPrimary, fontWeight = FontWeight.Bold)
                }
                Divider(modifier = Modifier.height(36.dp).width(1.dp), color = SurfaceCard)
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("TARGET DURATION", fontSize = 11.sp, color = TextSecondary, fontWeight = FontWeight.SemiBold)
                    Text("${task.durationMin}–${task.durationMax}s", fontSize = 16.sp, color = CyanAccent, fontWeight = FontWeight.Bold)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Protocol Procedure Instructions
        Text(
            text = "PROCEDURE PROTOCOL (${task.instructionVersion})",
            fontSize = 13.sp,
            fontWeight = FontWeight.Bold,
            color = TextSecondary,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Card(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = SurfaceDark),
            shape = RoundedCornerShape(12.dp)
        ) {
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                itemsIndexed(task.procedureSteps) { index, step ->
                    Row(verticalAlignment = Alignment.Top) {
                        Text(
                            text = "${index + 1}.",
                            color = CyanAccent,
                            fontWeight = FontWeight.Bold,
                            fontSize = 14.sp,
                            modifier = Modifier.width(28.dp)
                        )
                        Text(
                            text = step.replace(Regex("^\\d+\\.\\s*"), ""),
                            color = TextPrimary,
                            fontSize = 14.sp,
                            lineHeight = 20.sp
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Large CTA Button
        Button(
            onClick = onStartRecording,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            colors = ButtonDefaults.buttonColors(containerColor = CyanAccent),
            shape = RoundedCornerShape(12.dp)
        ) {
            Icon(Icons.Default.Videocam, contentDescription = null, tint = SpaceBackground)
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = "START RECORDING",
                color = SpaceBackground,
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
        }
    }
}
