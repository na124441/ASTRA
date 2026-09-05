package org.astra.collector.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface UploadDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUpload(upload: UploadRecordEntity)

    @Update
    suspend fun updateUpload(upload: UploadRecordEntity)

    @Query("SELECT * FROM upload_records WHERE uploadId = :uploadId LIMIT 1")
    suspend fun getUploadById(uploadId: String): UploadRecordEntity?

    @Query("SELECT * FROM upload_records WHERE status IN ('PENDING', 'UPLOADING', 'FAILED')")
    suspend fun getPendingUploads(): List<UploadRecordEntity>

    @Query("SELECT * FROM upload_records WHERE status = 'REMOTE_COMPLETE_LOCAL_DELETE_PENDING'")
    suspend fun getPendingDeletions(): List<UploadRecordEntity>

    @Query("SELECT * FROM upload_records WHERE taskId = :taskId ORDER BY createdAt DESC LIMIT 1")
    fun getUploadForTaskFlow(taskId: String): Flow<UploadRecordEntity?>
}
