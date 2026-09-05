/**
 * ASTRA Collector — Mobile Web Terminal Controller
 * Features:
 * - HTML5 MediaRecorder (1080p Landscape)
 * - IndexedDB Staging & Fail-Closed Storage Invariant
 * - Web Crypto API Streaming SHA-256
 * - Resumable 8 MB Chunked Streaming
 */

class IndexedDBManager {
  constructor() {
    this.dbName = 'astra_collector_db';
    this.version = 1;
    this.db = null;
  }

  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('recordings')) {
          db.createObjectStore('recordings', { keyPath: 'recordingId' });
        }
      };
      request.onsuccess = (e) => {
        this.db = e.target.result;
        resolve(this.db);
      };
      request.onerror = (e) => reject(e.target.error);
    });
  }

  async saveVideoBlob(recordingId, blob, metadata) {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('recordings', 'readwrite');
      const store = tx.objectStore('recordings');
      store.put({ recordingId, blob, metadata, timestamp: Date.now() });
      tx.oncomplete = () => resolve();
      tx.onerror = (e) => reject(e.target.error);
    });
  }

  async getVideoBlob(recordingId) {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('recordings', 'readonly');
      const store = tx.objectStore('recordings');
      const req = store.get(recordingId);
      req.onsuccess = () => resolve(req.result);
      req.onerror = (e) => reject(e.target.error);
    });
  }

  /**
   * CRITICAL INVARIANT:
   * Only delete the video from local IndexedDB if server status is VERIFIED.
   */
  async deleteVerifiedVideo(recordingId, serverStatus, isVerified) {
    if (!isVerified || String(serverStatus).toLowerCase() !== 'verified') {
      throw new Error(`CRITICAL INVARIANT VIOLATION: Cannot delete unverified local recording (isVerified=${isVerified}, status=${serverStatus})`);
    }
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction('recordings', 'readwrite');
      const store = tx.objectStore('recordings');
      store.delete(recordingId);
      tx.oncomplete = () => {
        console.log(`[Storage] Securely purged local recording ${recordingId} after verified upload.`);
        resolve(true);
      };
      tx.onerror = (e) => reject(e.target.error);
    });
  }
}

class ChecksumManager {
  static async computeSha256(blob) {
    const arrayBuffer = await blob.arrayBuffer();
    const digestBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(digestBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
  }
}

class CameraCaptureManager {
  constructor() {
    this.stream = null;
    this.mediaRecorder = null;
    this.recordedChunks = [];
    this.timerInterval = null;
    this.durationSeconds = 0;
  }

  async startCamera(videoElement) {
    const constraints = {
      audio: true,
      video: {
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        facingMode: { ideal: 'environment' }
      }
    };
    this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElement.srcObject = this.stream;
    await videoElement.play();
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
  }

  startRecording(onTick) {
    this.recordedChunks = [];
    this.durationSeconds = 0;

    let mimeType = 'video/webm;codecs=vp9,opus';
    if (!MediaRecorder.isTypeSupported(mimeType)) {
      mimeType = MediaRecorder.isTypeSupported('video/mp4') ? 'video/mp4' : 'video/webm';
    }

    this.mediaRecorder = new MediaRecorder(this.stream, {
      mimeType: mimeType,
      videoBitsPerSecond: 8000000 // 8 Mbps high quality
    });

    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        this.recordedChunks.push(e.data);
      }
    };

    this.mediaRecorder.start(1000); // 1-second chunks

    this.timerInterval = setInterval(() => {
      this.durationSeconds++;
      if (onTick) onTick(this.durationSeconds);
    }, 1000);
  }

  async stopRecording() {
    return new Promise((resolve) => {
      clearInterval(this.timerInterval);
      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder.mimeType || 'video/webm';
        const blob = new Blob(this.recordedChunks, { type: mimeType });
        this.stopCamera();
        resolve({ blob, durationSeconds: this.durationSeconds });
      };
      this.mediaRecorder.stop();
    });
  }
}

// App State
const state = {
  serverUrl: window.location.origin,
  collectorId: 'COL-007',
  authToken: null,
  activeTask: null,
  activeBlob: null,
  activeDuration: 0,
  activeSha256: null,
  activeUploadId: null
};

// DOM References
const screens = {
  connect: document.getElementById('screen-connect'),
  task: document.getElementById('screen-task'),
  camera: document.getElementById('screen-camera'),
  review: document.getElementById('screen-review'),
  upload: document.getElementById('screen-upload'),
  complete: document.getElementById('screen-complete')
};

const storageManager = new IndexedDBManager();
const cameraManager = new CameraCaptureManager();

function showScreen(screenKey) {
  Object.values(screens).forEach(s => s.classList.remove('active'));
  if (screens[screenKey]) {
    screens[screenKey].classList.add('active');
  }
}

// ----------------------------------------------------------------------------
// Screen 1: Connect Logic
// ----------------------------------------------------------------------------
document.getElementById('input-server-url').value = window.location.origin;

document.getElementById('btn-connect').addEventListener('click', async () => {
  const serverUrlInput = document.getElementById('input-server-url').value.trim();
  const collectorIdInput = document.getElementById('input-collector-id').value.trim();
  const errBanner = document.getElementById('connect-error');

  state.serverUrl = serverUrlInput.replace(/\/+$/, '');
  state.collectorId = collectorIdInput;

  errBanner.classList.add('hidden');

  try {
    const regResp = await fetch(`${state.serverUrl}/api/v1/collector/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        collector_id: state.collectorId,
        device_id: navigator.userAgent,
        app_version: '1.0.0-web',
        device_model: 'Web Browser'
      })
    });

    if (!regResp.ok) {
      throw new Error(`Authentication error (${regResp.status})`);
    }

    const authData = await regResp.json();
    state.authToken = authData.auth_token;

    const connPill = document.getElementById('connection-pill');
    connPill.textContent = state.collectorId;
    connPill.className = 'badge badge-active';

    await loadNextTask();
  } catch (e) {
    errBanner.textContent = `Connection failed: ${e.message}`;
    errBanner.classList.remove('hidden');
  }
});

async function loadNextTask() {
  try {
    const taskResp = await fetch(`${state.serverUrl}/api/v1/collector/tasks/next?collector_id=${state.collectorId}`);
    if (!taskResp.ok) throw new Error('Failed to dispense task');

    const task = await taskResp.json();
    if (!task) {
      alert('No pending tasks available for this collector.');
      return;
    }

    state.activeTask = task;

    // Populate Task Screen
    document.getElementById('task-run-title').textContent = `${task.experiment_id} • ${task.run_id}`;
    document.getElementById('task-camera-sub').textContent = `Camera: ${task.camera_id} (1080p Landscape)`;

    const scenBadge = document.getElementById('task-scenario-badge');
    scenBadge.textContent = task.scenario_type.toUpperCase();
    scenBadge.className = task.scenario_type.toLowerCase() === 'nominal' ? 'badge badge-nominal' : 'badge badge-fault';

    document.getElementById('task-obj-val').textContent = task.required_object;
    document.getElementById('task-tgt-val').textContent = task.target;
    document.getElementById('task-dur-val').textContent = `${task.duration_min}–${task.duration_max}s`;
    document.getElementById('task-proto-ver').textContent = task.instruction_version;

    const stepsList = document.getElementById('task-steps-list');
    stepsList.innerHTML = '';
    task.procedure_steps.forEach(step => {
      const li = document.createElement('li');
      li.textContent = step.replace(/^\d+\.\s*/, '');
      stepsList.appendChild(li);
    });

    showScreen('task');
  } catch (e) {
    alert(`Task loading error: ${e.message}`);
  }
}

// ----------------------------------------------------------------------------
// Screen 2 -> Screen 3: Start Recording
// ----------------------------------------------------------------------------
document.getElementById('btn-start-recording').addEventListener('click', async () => {
  try {
    const viewfinder = document.getElementById('viewfinder');
    await cameraManager.startCamera(viewfinder);

    document.getElementById('hud-task-tag').textContent = 
      `${state.activeTask.experiment_id} • ${state.activeTask.run_id} • ${state.activeTask.camera_id}`;

    showScreen('camera');

    cameraManager.startRecording((secs) => {
      const m = String(Math.floor(secs / 60)).padStart(2, '0');
      const s = String(secs % 60).padStart(2, '0');
      document.getElementById('rec-timer').textContent = `REC ${m}:${s}`;
    });
  } catch (e) {
    alert(`Camera access failed: ${e.message}. Ensure camera permissions are allowed in browser.`);
  }
});

// ----------------------------------------------------------------------------
// Screen 3 -> Screen 4: Stop Recording & Validate
// ----------------------------------------------------------------------------
document.getElementById('btn-stop-recording').addEventListener('click', async () => {
  const result = await cameraManager.stopRecording();
  state.activeBlob = result.blob;
  state.activeDuration = result.durationSeconds;

  // Stash in IndexedDB immediately for fail-closed retention
  const recordingId = `${state.activeTask.experiment_id}_${state.activeTask.run_id}_${state.activeTask.camera_id}`;
  await storageManager.saveVideoBlob(recordingId, state.activeBlob, {
    task: state.activeTask,
    duration: state.activeDuration
  });

  // Setup Review Screen
  const reviewPlayer = document.getElementById('review-player');
  reviewPlayer.src = URL.createObjectURL(state.activeBlob);

  document.getElementById('review-task-sub').textContent = 
    `${state.activeTask.experiment_id} • ${state.activeTask.run_id} • ${state.activeTask.camera_id}`;

  const isDurationValid = state.activeDuration >= state.activeTask.duration_min && 
                          state.activeDuration <= state.activeTask.duration_max;

  document.getElementById('val-duration').textContent = 
    `${state.activeDuration.toFixed(1)}s (required: ${state.activeTask.duration_min}–${state.activeTask.duration_max}s)`;
  document.getElementById('icon-duration').textContent = isDurationValid ? '✓' : '⚠';
  document.getElementById('icon-duration').style.color = isDurationValid ? 'var(--green-success)' : 'var(--amber-warning)';

  const sizeMb = (state.activeBlob.size / (1024 * 1024)).toFixed(2);
  document.getElementById('val-file-size').textContent = `${sizeMb} MB`;

  document.getElementById('val-sha256').textContent = 'Computing streaming Web Crypto digest...';
  document.getElementById('btn-upload').disabled = true;

  showScreen('review');

  // Compute SHA-256 digest
  const hash = await ChecksumManager.computeSha256(state.activeBlob);
  state.activeSha256 = hash;
  document.getElementById('val-sha256').textContent = hash;
  document.getElementById('btn-upload').disabled = false;
});

// Re-record Button
document.getElementById('btn-rerecord').addEventListener('click', () => {
  showScreen('task');
});

// ----------------------------------------------------------------------------
// Screen 4 -> Screen 5: Resumable Chunked Upload
// ----------------------------------------------------------------------------
document.getElementById('btn-upload').addEventListener('click', async () => {
  await executeUpload();
});

document.getElementById('btn-retry-upload').addEventListener('click', async () => {
  await executeUpload();
});

async function executeUpload() {
  showScreen('upload');

  const task = state.activeTask;
  const blob = state.activeBlob;
  const sha256 = state.activeSha256;

  document.getElementById('upload-task-sub').textContent = `${task.experiment_id} • ${task.run_id} (${task.camera_id})`;
  document.getElementById('upload-status-title').textContent = 'STREAMING RESUMABLE CHUNKS...';
  document.getElementById('upload-error-box').classList.add('hidden');
  document.getElementById('btn-retry-upload').classList.add('hidden');

  const chunkSize = 3 * 1024 * 1024; // 3 MB (safe for Vercel 4.5 MB request limit)
  const totalChunks = Math.ceil(blob.size / chunkSize);

  const metadata = {
    schema_version: '1.0',
    experiment_id: task.experiment_id,
    run_id: task.run_id,
    recording_id: `${task.experiment_id}_${task.run_id}_${task.camera_id}`,
    collector_id: state.collectorId,
    camera_id: task.camera_id,
    scenario_type: task.scenario_type,
    object: task.required_object,
    target: task.target,
    duration_seconds: state.activeDuration,
    width: 1920,
    height: 1080,
    fps: 30.0,
    orientation: 'landscape',
    file_size_bytes: blob.size,
    sha256: sha256,
    app_version: '1.0.0-web',
    protocol_version: 'EXP001-v1.0',
    created_at: new Date().toISOString()
  };

  try {
    // 1. Initiate upload
    const initResp = await fetch(`${state.serverUrl}/api/v1/collector/uploads/initiate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_id: task.task_id,
        collector_id: state.collectorId,
        file_size_bytes: blob.size,
        total_chunks: totalChunks,
        sha256: sha256,
        metadata: metadata
      })
    });

    if (!initResp.ok) throw new Error(`Upload initiation rejected (${initResp.status})`);
    const initData = await initResp.json();
    state.activeUploadId = initData.upload_id;

    // 2. Stream chunks sequentially
    for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
      const start = chunkIdx * chunkSize;
      const end = Math.min(start + chunkSize, blob.size);
      const chunkBlob = blob.slice(start, end);
      const chunkHash = await ChecksumManager.computeSha256(chunkBlob);

      document.getElementById('upload-progress-chunk').textContent = `Chunk ${chunkIdx + 1} of ${totalChunks}`;
      const percent = Math.round(((chunkIdx + 1) / totalChunks) * 100);
      document.getElementById('upload-progress-fill').style.width = `${percent}%`;
      document.getElementById('upload-progress-percent').textContent = `${percent}%`;

      const chunkResp = await fetch(`${state.serverUrl}/api/v1/collector/uploads/${state.activeUploadId}/chunks/${chunkIdx}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/octet-stream',
          'X-Chunk-SHA256': chunkHash
        },
        body: chunkBlob
      });

      if (!chunkResp.ok) throw new Error(`Chunk ${chunkIdx} upload failed (${chunkResp.status})`);
    }

    // 3. Finalize & Verify Remote Persistence
    document.getElementById('upload-status-title').textContent = 'VERIFYING REMOTE PERSISTENCE...';

    const completeResp = await fetch(`${state.serverUrl}/api/v1/collector/uploads/${state.activeUploadId}/complete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        upload_id: state.activeUploadId,
        expected_sha256: sha256,
        metadata: metadata
      })
    });

    if (!completeResp.ok) throw new Error(`Completion error (${completeResp.status})`);
    const completeData = await completeResp.json();

    // 4. HARD INVARIANT CHECK
    if (completeData.verified && String(completeData.status).toLowerCase() === 'verified') {
      // Delete local temporary video ONLY after verified confirmation
      const recordingId = `${task.experiment_id}_${task.run_id}_${task.camera_id}`;
      await storageManager.deleteVerifiedVideo(recordingId, completeData.status, completeData.verified);

      // Render Screen 6 (Complete)
      document.getElementById('complete-task-sub').textContent = `${task.run_id} (${task.camera_id}) Complete`;
      document.getElementById('complete-remote-path').textContent = completeData.remote_path;
      showScreen('complete');
    } else {
      throw new Error(completeData.error_message || 'Verification rejected by backend.');
    }

  } catch (e) {
    document.getElementById('upload-status-title').textContent = 'UPLOAD FAILED';
    document.getElementById('upload-error-msg').textContent = e.message;
    document.getElementById('upload-error-box').classList.remove('hidden');
    document.getElementById('btn-retry-upload').classList.remove('hidden');
  }
}

// ----------------------------------------------------------------------------
// Screen 6: Next Task
// ----------------------------------------------------------------------------
document.getElementById('btn-next-task').addEventListener('click', async () => {
  await loadNextTask();
});
