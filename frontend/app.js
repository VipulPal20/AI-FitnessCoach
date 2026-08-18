/**
 * Form Coach AI — Main Client Logic
 * Handles camera capture (WebRTC), drag & drop upload, pose overlay rendering,
 * and biomechanical analysis report.
 */

// Global App State
const state = {
  activeTab: 'camera',
  webcamStream: null,
  facingMode: 'user',
  timerEnabled: true,
  countdown: 3,
  isCapturing: false,
  selectedFile: null,
  analysisResult: null,
  sourceImageElement: null
};

// DOM Elements
const DOM = {
  // Tabs & Nav
  healthStatus: document.querySelector('#healthStatus'),
  healthBadge: document.querySelector('#healthBadge'),
  tabCamera: document.querySelector('#tabCamera'),
  tabUpload: document.querySelector('#tabUpload'),
  cameraView: document.querySelector('#cameraView'),
  uploadView: document.querySelector('#uploadView'),

  // Camera Elements
  webcamVideo: document.querySelector('#webcamVideo'),
  cameraGuide: document.querySelector('#cameraGuide'),
  countdownOverlay: document.querySelector('#countdownOverlay'),
  cameraError: document.querySelector('#cameraError'),
  cameraErrMsg: document.querySelector('#cameraErrMsg'),
  retryCameraBtn: document.querySelector('#retryCameraBtn'),
  shutterBtn: document.querySelector('#shutterBtn'),
  toggleTimerBtn: document.querySelector('#toggleTimerBtn'),
  timerStatus: document.querySelector('#timerStatus'),
  switchCamBtn: document.querySelector('#switchCamBtn'),

  // Upload Elements
  dropZone: document.querySelector('#dropZone'),
  fileInput: document.querySelector('#fileInput'),
  browseBtn: document.querySelector('#browseBtn'),
  sampleChips: document.querySelectorAll('.sample-chip'),

  // Selection & Actions
  selectedInfo: document.querySelector('#selectedInfo'),
  selectedThumb: document.querySelector('#selectedThumb'),
  selectedName: document.querySelector('#selectedName'),
  selectedMeta: document.querySelector('#selectedMeta'),
  removeSelectedBtn: document.querySelector('#removeSelectedBtn'),
  confidence: document.querySelector('#confidence'),
  confidenceValue: document.querySelector('#confidenceValue'),
  analyseButton: document.querySelector('#analyseButton'),

  // Display States
  loadingState: document.querySelector('#loadingState'),
  errorState: document.querySelector('#errorState'),
  errorText: document.querySelector('#errorText'),
  tryAgainBtn: document.querySelector('#tryAgainBtn'),
  report: document.querySelector('#report'),

  // Report Elements
  preview: document.querySelector('#preview'),
  poseCanvas: document.querySelector('#poseCanvas'),
  scoreValue: document.querySelector('#scoreValue'),
  scoreRingProgress: document.querySelector('#scoreRingProgress'),
  depthChip: document.querySelector('#depthChip'),
  exerciseBadge: document.querySelector('#exerciseBadge'),
  leftAngle: document.querySelector('#leftAngle'),
  rightAngle: document.querySelector('#rightAngle'),
  kneeDiff: document.querySelector('#kneeDiff'),
  torsoLean: document.querySelector('#torsoLean'),
  kneeTracking: document.querySelector('#kneeTracking'),
  poseConfidence: document.querySelector('#poseConfidence'),
  keypointsCount: document.querySelector('#keypointsCount'),
  feedbackList: document.querySelector('#feedbackList'),
  downloadAnnotatedBtn: document.querySelector('#downloadAnnotatedBtn'),

  // Canvas Toggles
  toggleSkeleton: document.querySelector('#toggleSkeleton'),
  toggleJoints: document.querySelector('#toggleJoints'),
  toggleLabels: document.querySelector('#toggleLabels')
};

// COCO Skeleton Connections (17 keypoints)
const COCO_PAIRS = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle']
];

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  checkBackendHealth();
  setupEventListeners();
  initCamera();
});

// --- Health Check ---
async function checkBackendHealth() {
  try {
    const res = await fetch('/api/v1/health');
    const data = await res.json();
    if (res.ok && data.status === 'ok') {
      DOM.healthStatus.textContent = `Online (${data.model})`;
      DOM.healthBadge.style.color = 'var(--accent-emerald)';
    } else {
      DOM.healthStatus.textContent = `Degraded: ${data.message || 'Model loading...'}`;
      DOM.healthBadge.style.color = 'var(--accent-amber)';
    }
  } catch (err) {
    DOM.healthStatus.textContent = 'API Offline';
    DOM.healthBadge.style.color = 'var(--accent-rose)';
  }
}

// --- Event Listeners Setup ---
function setupEventListeners() {
  // Tab Switching
  DOM.tabCamera.addEventListener('click', () => switchTab('camera'));
  DOM.tabUpload.addEventListener('click', () => switchTab('upload'));

  // Camera Controls
  DOM.shutterBtn.addEventListener('click', handleShutterClick);
  DOM.toggleTimerBtn.addEventListener('click', toggleTimer);
  DOM.switchCamBtn.addEventListener('click', switchCamera);
  DOM.retryCameraBtn.addEventListener('click', initCamera);

  // Upload File Picker
  DOM.browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    DOM.fileInput.click();
  });
  DOM.dropZone.addEventListener('click', () => DOM.fileInput.click());

  DOM.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0], 'upload');
    }
  });

  // Global & DropZone Drag and Drop Handlers
  setupDragAndDrop();

  // Sample Images
  DOM.sampleChips.forEach(chip => {
    chip.addEventListener('click', async (e) => {
      e.stopPropagation();
      const src = chip.getAttribute('data-src');
      try {
        const res = await fetch(src);
        const blob = await res.blob();
        const file = new File([blob], 'sample_squat.webp', { type: blob.type || 'image/webp' });
        handleFileSelected(file, 'sample');
      } catch (err) {
        showError('Unable to load sample image.');
      }
    });
  });

  // Selection & Actions
  DOM.removeSelectedBtn.addEventListener('click', clearSelection);
  DOM.confidence.addEventListener('input', updateConfidenceLabel);
  DOM.analyseButton.addEventListener('click', runFormAnalysis);
  DOM.tryAgainBtn.addEventListener('click', resetState);
  DOM.downloadAnnotatedBtn.addEventListener('click', downloadAnnotatedSnapshot);

  // Canvas View Toggles
  [DOM.toggleSkeleton, DOM.toggleJoints, DOM.toggleLabels].forEach(input => {
    input.addEventListener('change', () => {
      if (state.analysisResult && state.sourceImageElement) {
        renderPoseCanvas(state.sourceImageElement, state.analysisResult.people[0]?.landmarks || {});
      }
    });
  });

  window.addEventListener('resize', () => {
    if (state.analysisResult && state.sourceImageElement) {
      renderPoseCanvas(state.sourceImageElement, state.analysisResult.people[0]?.landmarks || {});
    }
  });
}

// --- Bulletproof Drag & Drop Setup ---
function setupDragAndDrop() {
  // Prevent browser default behavior (opening image in new tab)
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    window.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
  });

  // Visual cues for drag hover
  ['dragenter', 'dragover'].forEach(eventName => {
    window.addEventListener(eventName, () => {
      DOM.dropZone.classList.add('dragging');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    window.addEventListener(eventName, () => {
      DOM.dropZone.classList.remove('dragging');
    }, false);
  });

  // Process dropped file
  window.addEventListener('drop', (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      if (state.activeTab !== 'upload') {
        switchTab('upload');
      }
      handleFileSelected(file, 'drag & drop');
    }
  }, false);
}

// --- Tab Controller ---
function switchTab(tab) {
  state.activeTab = tab;
  if (tab === 'camera') {
    DOM.tabCamera.classList.add('active');
    DOM.tabUpload.classList.remove('active');
    DOM.cameraView.classList.add('active');
    DOM.uploadView.classList.remove('active');
    initCamera();
  } else {
    DOM.tabUpload.classList.add('active');
    DOM.tabCamera.classList.remove('active');
    DOM.uploadView.classList.add('active');
    DOM.cameraView.classList.remove('active');
    stopCamera();
  }
}

// --- WebRTC Camera Controller ---
async function initCamera() {
  DOM.cameraError.hidden = true;
  stopCamera();
  try {
    const constraints = {
      video: {
        facingMode: state.facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };
    state.webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
    DOM.webcamVideo.srcObject = state.webcamStream;
    DOM.cameraGuide.hidden = false;

    // Check multiple devices for camera switcher button
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    DOM.switchCamBtn.hidden = videoDevices.length <= 1;
  } catch (err) {
    console.error('Camera Access Error:', err);
    DOM.cameraErrMsg.textContent = 'Camera access denied or unavailable on this device.';
    DOM.cameraError.hidden = false;
    DOM.cameraGuide.hidden = true;
  }
}

function stopCamera() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
  DOM.webcamVideo.srcObject = null;
}

function switchCamera() {
  state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
  initCamera();
}

function toggleTimer() {
  state.timerEnabled = !state.timerEnabled;
  DOM.timerStatus.textContent = state.timerEnabled ? 'Timer: 3s' : 'Timer: Off';
  DOM.toggleTimerBtn.style.color = state.timerEnabled ? 'var(--accent-cyan)' : 'var(--text-muted)';
}

function handleShutterClick() {
  if (state.isCapturing) return;
  if (state.timerEnabled) {
    startCountdown();
  } else {
    takeSnapshot();
  }
}

function startCountdown() {
  state.isCapturing = true;
  let seconds = 3;
  DOM.countdownOverlay.textContent = seconds;
  DOM.countdownOverlay.hidden = false;

  const interval = setInterval(() => {
    seconds--;
    if (seconds > 0) {
      DOM.countdownOverlay.textContent = seconds;
    } else {
      clearInterval(interval);
      DOM.countdownOverlay.hidden = true;
      state.isCapturing = false;
      takeSnapshot();
    }
  }, 1000);
}

function takeSnapshot() {
  const video = DOM.webcamVideo;
  if (!video.videoWidth || !video.videoHeight) return;

  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = video.videoWidth;
  offscreenCanvas.height = video.videoHeight;
  const ctx = offscreenCanvas.getContext('2d');
  
  // If front camera, mirror snapshot to match preview
  if (state.facingMode === 'user') {
    ctx.translate(offscreenCanvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0);

  offscreenCanvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], `camera_photo_${Date.now()}.jpg`, { type: 'image/jpeg' });
    handleFileSelected(file, 'camera');
  }, 'image/jpeg', 0.95);
}

// --- Selection Handler ---
function handleFileSelected(file, source) {
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showError('Unsupported file type. Please choose a JPEG, PNG, or WEBP photo.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError('File size exceeds maximum limit of 10 MB.');
    return;
  }

  state.selectedFile = file;
  DOM.selectedName.textContent = file.name;
  DOM.selectedMeta.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB · ${source.toUpperCase()}`;

  const objectUrl = URL.createObjectURL(file);
  DOM.selectedThumb.src = objectUrl;
  DOM.selectedInfo.hidden = false;
  DOM.analyseButton.disabled = false;

  resetState(false);
}

function clearSelection() {
  state.selectedFile = null;
  DOM.fileInput.value = '';
  DOM.selectedInfo.hidden = true;
  DOM.analyseButton.disabled = true;
  resetState(true);
}

function updateConfidenceLabel() {
  const v = parseFloat(DOM.confidence.value);
  const modeText = v < 0.3 ? 'Fast' : v > 0.55 ? 'Precise' : 'Balanced';
  DOM.confidenceValue.textContent = `${v.toFixed(2)} — ${modeText}`;
}

function resetState(hideReport = true) {
  DOM.loadingState.hidden = true;
  DOM.errorState.hidden = true;
  if (hideReport) {
    DOM.report.hidden = true;
  }
}

function showError(msg) {
  DOM.loadingState.hidden = true;
  DOM.report.hidden = true;
  DOM.errorText.textContent = msg;
  DOM.errorState.hidden = false;
}

// --- Form Analysis API Call ---
async function runFormAnalysis() {
  if (!state.selectedFile) return;

  DOM.report.hidden = true;
  DOM.errorState.hidden = true;
  DOM.loadingState.hidden = false;
  DOM.analyseButton.disabled = true;

  const formData = new FormData();
  formData.append('file', state.selectedFile);

  try {
    const response = await fetch(`/api/v1/analyze?confidence=${DOM.confidence.value}`, {
      method: 'POST',
      body: formData
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || result.message || 'Analysis request failed.');
    }

    if (!result.data || !result.data.detected) {
      showError(result.data?.message || 'No full-body person pose was detected. Please try a clearer, wider photo.');
      return;
    }

    state.analysisResult = result.data;
    renderAnalysisReport(result.data);

  } catch (err) {
    showError(err.message || 'Connection interrupted during analysis. Please try again.');
  } finally {
    DOM.loadingState.hidden = true;
    DOM.analyseButton.disabled = false;
  }
}

// --- Render Report Dashboard ---
function renderAnalysisReport(data) {
  const person = data.people[0];
  const metrics = person.metrics;
  const angles = metrics.knee_angles || {};
  const posture = metrics.posture || {};

  // Setup preview image
  const imgUrl = URL.createObjectURL(state.selectedFile);
  DOM.preview.src = imgUrl;
  DOM.preview.onload = () => {
    state.sourceImageElement = DOM.preview;
    renderPoseCanvas(DOM.preview, person.landmarks);
  };

  // Score Calculation
  const formScore = calculateFormScore(metrics.squat_depth, person.confidence, posture);
  DOM.scoreValue.textContent = formScore;
  updateScoreRing(formScore);

  // Depth & Exercise Badges
  const depth = metrics.squat_depth || 'unknown';
  DOM.depthChip.textContent = `${depth} depth`;
  DOM.depthChip.className = `chip-depth ${depth}`;
  
  const exerciseLabel = metrics.exercise?.label ? capitalize(metrics.exercise.label) : 'Squat / Pose';
  DOM.exerciseBadge.textContent = `Exercise: ${exerciseLabel}`;

  // Angles & Posture Cards
  DOM.leftAngle.textContent = angles.left !== undefined ? `${angles.left}°` : 'N/A';
  DOM.rightAngle.textContent = angles.right !== undefined ? `${angles.right}°` : 'N/A';

  if (posture.knee_angle_difference !== null && posture.knee_angle_difference !== undefined) {
    DOM.kneeDiff.textContent = `${posture.knee_angle_difference}°`;
    DOM.kneeDiffSub.textContent = posture.knee_angle_difference > 15 ? 'High Imbalance' : 'Balanced';
    DOM.kneeDiffSub.style.color = posture.knee_angle_difference > 15 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
  } else {
    DOM.kneeDiff.textContent = 'N/A';
    DOM.kneeDiffSub.textContent = 'Need both legs visible';
  }

  if (posture.torso_lean_degrees !== null && posture.torso_lean_degrees !== undefined) {
    DOM.torsoLean.textContent = `${posture.torso_lean_degrees}°`;
    DOM.torsoLeanSub.textContent = posture.torso_lean_degrees > 45 ? 'Forward Lean' : 'Upright Chest';
    DOM.torsoLeanSub.style.color = posture.torso_lean_degrees > 45 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
  } else {
    DOM.torsoLean.textContent = 'N/A';
    DOM.torsoLeanSub.textContent = 'Need torso view';
  }

  DOM.kneeTracking.textContent = posture.knee_tracking === 'aligned' ? 'Aligned' : posture.knee_tracking === 'check_alignment' ? 'Valgus Risk' : 'N/A';
  DOM.kneeTracking.style.color = posture.knee_tracking === 'aligned' ? 'var(--accent-emerald)' : posture.knee_tracking === 'check_alignment' ? 'var(--accent-amber)' : 'var(--text-dim)';

  DOM.poseConfidence.textContent = `${Math.round(person.confidence * 100)}%`;
  DOM.keypointsCount.textContent = `${metrics.visible_keypoints} / 17 keypoints`;

  // Feedback Bullet List
  DOM.feedbackList.innerHTML = '';
  if (person.feedback && person.feedback.length > 0) {
    person.feedback.forEach(item => {
      const li = document.createElement('li');
      li.textContent = item;
      DOM.feedbackList.appendChild(li);
    });
  } else {
    const li = document.createElement('li');
    li.textContent = 'Great overall form! Maintain your balance and posture.';
    DOM.feedbackList.appendChild(li);
  }

  DOM.report.hidden = false;
  DOM.report.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Score Ring Animation (264 total circumference)
function updateScoreRing(score) {
  const maxDash = 264;
  const offset = maxDash - (maxDash * (score / 100));
  DOM.scoreRingProgress.style.strokeDashoffset = offset;
  
  if (score >= 85) {
    DOM.scoreRingProgress.style.stroke = 'var(--accent-emerald)';
  } else if (score >= 65) {
    DOM.scoreRingProgress.style.stroke = 'var(--accent-cyan)';
  } else {
    DOM.scoreRingProgress.style.stroke = 'var(--accent-amber)';
  }
}

function calculateFormScore(depth, confidence, posture) {
  const baseMap = { deep: 95, parallel: 82, shallow: 65, unknown: 50 };
  let score = baseMap[depth] || 50;

  // Deduct points for high knee asymmetry or extreme forward torso lean
  if (typeof posture.knee_angle_difference === 'number' && posture.knee_angle_difference > 15) {
    score -= 10;
  }
  if (typeof posture.torso_lean_degrees === 'number' && posture.torso_lean_degrees > 50) {
    score -= 10;
  }

  // Weight confidence factor
  const finalScore = Math.round(score * 0.8 + confidence * 20);
  return Math.max(10, Math.min(100, finalScore));
}

// --- Canvas Pose Skeleton Overlay ---
function renderPoseCanvas(imgElement, landmarks) {
  const canvas = DOM.poseCanvas;
  const rect = imgElement.getBoundingClientRect();
  const naturalW = imgElement.naturalWidth || rect.width;
  const naturalH = imgElement.naturalHeight || rect.height;

  if (!rect.width || !rect.height) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const scaleX = rect.width / naturalW;
  const scaleY = rect.height / naturalH;

  // 1. Draw Skeleton Lines
  if (DOM.toggleSkeleton.checked) {
    ctx.strokeStyle = '#00ff87';
    ctx.lineWidth = 3;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#00ff87';

    COCO_PAIRS.forEach(([p1, p2]) => {
      const pt1 = landmarks[p1];
      const pt2 = landmarks[p2];
      if (pt1 && pt2 && pt1.confidence > 0.25 && pt2.confidence > 0.25) {
        ctx.beginPath();
        ctx.moveTo(pt1.x * scaleX, pt1.y * scaleY);
        ctx.lineTo(pt2.x * scaleX, pt2.y * scaleY);
        ctx.stroke();
      }
    });
  }

  // 2. Draw Joint Markers
  if (DOM.toggleJoints.checked) {
    Object.entries(landmarks).forEach(([name, pt]) => {
      if (pt.confidence > 0.25) {
        const x = pt.x * scaleX;
        const y = pt.y * scaleY;

        // Outer Glow Ring
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 240, 255, 0.4)';
        ctx.fill();

        // Inner Dot
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    });
  }

  // 3. Draw Knee Angle Labels
  if (DOM.toggleLabels.checked) {
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00f0ff';
    ctx.shadowBlur = 6;
    ctx.shadowColor = '#000000';

    ['left_knee', 'right_knee'].forEach(kneeName => {
      const pt = landmarks[kneeName];
      if (pt && pt.confidence > 0.25) {
        const label = kneeName.includes('left') ? 'L Knee' : 'R Knee';
        ctx.fillText(label, pt.x * scaleX + 10, pt.y * scaleY + 4);
      }
    });
  }
}

// --- Download Annotated Photo Feature ---
function downloadAnnotatedSnapshot() {
  if (!state.sourceImageElement) return;

  const exportCanvas = document.createElement('canvas');
  const img = state.sourceImageElement;
  const w = img.naturalWidth;
  const h = img.naturalHeight;

  exportCanvas.width = w;
  exportCanvas.height = h;
  const ctx = exportCanvas.getContext('2d');

  // Draw source image
  ctx.drawImage(img, 0, 0, w, h);

  // Draw overlay landmarks directly at 1:1 scale
  const landmarks = state.analysisResult?.people[0]?.landmarks || {};

  // Lines
  ctx.strokeStyle = '#00ff87';
  ctx.lineWidth = Math.max(3, Math.round(w / 300));
  COCO_PAIRS.forEach(([p1, p2]) => {
    const pt1 = landmarks[p1];
    const pt2 = landmarks[p2];
    if (pt1 && pt2 && pt1.confi