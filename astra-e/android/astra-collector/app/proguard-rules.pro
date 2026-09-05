# ASTRA Collector Proguard Rules
-keepattributes *Annotation*
-keepclassmembers class * {
    @androidx.room.* <methods>;
}
