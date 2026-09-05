package org.astra.collector.data.local

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

@Dao
interface TaskDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTask(task: TaskEntity)

    @Update
    suspend fun updateTask(task: TaskEntity)

    @Query("SELECT * FROM tasks WHERE taskId = :taskId LIMIT 1")
    suspend fun getTaskById(taskId: String): TaskEntity?

    @Query("SELECT * FROM tasks WHERE status IN ('assigned', 'in_progress') ORDER BY assignedAt DESC LIMIT 1")
    fun getActiveTaskFlow(): Flow<TaskEntity?>

    @Query("SELECT * FROM tasks WHERE status IN ('assigned', 'in_progress') ORDER BY assignedAt DESC LIMIT 1")
    suspend fun getActiveTask(): TaskEntity?
}
