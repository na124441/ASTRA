package org.astra.collector.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val DarkColorScheme = darkColorScheme(
    primary = CyanAccent,
    secondary = CyanAccentDim,
    background = SpaceBackground,
    surface = SurfaceDark,
    onPrimary = SpaceBackground,
    onBackground = TextPrimary,
    onSurface = TextPrimary
)

@Composable
fun AstraCollectorTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = DarkColorScheme,
        content = content
    )
}
