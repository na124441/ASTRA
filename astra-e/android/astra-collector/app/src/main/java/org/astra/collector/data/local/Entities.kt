package org.astra.collector.data.local

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "tasks")
data class TaskEntity(
    @PrimaryKey val taskId: String,
    val experimentId: String,
    val runId: String,
    val cameraId: String,
    val scenarioType: String,
    val requiredObject: String,
    val target: String,
    val durationMin: Int,
    val durationMax: Int,
    val procedureStepsJson: String,
    val status: String,
    val assignedAt: Long
)

@Entity(tableName = "upload_records")
data class UploadRecordEntity(
    @PrimaryKey val uploadId: String,
    val taskId: String,
    val localFilePath: String,
    val sha256: String,
    val fileSizeBytes: Long,
    val status: String, // PENDING, UPLOADING, VERIFYING, COMPLETED, FAILED, REMOTE_COMPLETE_LOCAL_DELETE_PENDING
    val remotePath: String?,
    val retryCount: Int,
    val createdAt: Long,
    val updatedAt: Long
)

@Entity(tableName = "device_profile")
data class DeviceEntity(
    @PrimaryKey val collectorId: String,
    val deviceId: String,
    val authToken: String,
    val serverUrl: String,
    val isRegistered: Boolean,
    val lastCheckIn: Long
)
