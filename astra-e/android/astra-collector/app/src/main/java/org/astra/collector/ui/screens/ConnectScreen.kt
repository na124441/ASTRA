package org.astra.collector.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.astra.collector.ui.theme.*

@Composable
fun ConnectScreen(
    onConnected: (serverUrl: String, collectorId: String) -> Unit
) {
    var serverUrl by remember { mutableStateOf("http://10.0.2.2:8000") }
    var collectorId by remember { mutableStateOf("COL-007") }
    var isLoading by remember { mutableStateOf(false) }

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
                Text(
                    text = "ASTRA COLLECTOR",
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold,
                    color = CyanAccent,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    text = "Bhartiya Antariksh Station • SIH 26174",
                    fontSize = 12.sp,
                    color = TextSecondary,
                    modifier = Modifier.padding(top = 4.dp, bottom = 24.dp)
                )

                OutlinedTextField(
                    value = serverUrl,
                    onValueChange = { serverUrl = it },
                    label = { Text("Upload API Server URL", color = TextSecondary) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth().padding(bottom = 16.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedBorderColor = CyanAccent,
                        unfocusedBorderColor = SurfaceCard
                    )
                )

                OutlinedTextField(
                    value = collectorId,
                    onValueChange = { collectorId = it },
                    label = { Text("Synthesizer Collector ID", color = TextSecondary) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth().padding(bottom = 28.dp),
                    colors = OutlinedTextFieldDefaults.colors(
                        focusedTextColor = TextPrimary,
                        unfocusedTextColor = TextPrimary,
                        focusedBorderColor = CyanAccent,
                        unfocusedBorderColor = SurfaceCard
                    )
                )

                Button(
                    onClick = {
                        isLoading = true
                        onConnected(serverUrl.trim(), collectorId.trim())
                    },
                    enabled = serverUrl.isNotBlank() && collectorId.isNotBlank() && !isLoading,
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = CyanAccent),
                    shape = RoundedCornerShape(8.dp)
                ) {
                    if (isLoading) {
                        CircularProgressIndicator(color = SpaceBackground, modifier = Modifier.size(24.dp))
                    } else {
                        Text(
                            text = "CONNECT TERMINAL",
                            color = SpaceBackground,
                            fontWeight = FontWeight.Bold,
                            fontSize = 16.sp
                        )
                    }
                }
            }
        }
    }
}
