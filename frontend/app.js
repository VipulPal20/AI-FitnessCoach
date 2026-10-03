/**
 * PulseAI — Fitness Vision Trainer Client Application
 * Real-Time Exercise Recognition, Kinematic Angles & Repetition Counter
 * Architecture inspired by Chris Prasanna (Exercise_Recognition_AI).
 */

const state = {
  activeMode: 'live',
  isStreaming: false,
  webcamStream: null,
  facingMode: 'user',
  audioEnabled: true,
  fps: 0,
  frameCount: 0,
  lastFpsTime: performance.now(),
  isProcessingFrame: false,
  sendIntervalMs: 80,
  intervalTimer: null,
  videoPlaybackTimer: null,

  // Workout Session State
  sessionSeconds: 0,
  sessionTimerId: null,
  totalReps: 0,
  targets: { curl: 12, press: 10, squat: 15 },

  // Exercise Library State
  library: {
    page: 1,
    perPage: 24,
    totalPages: 1,
    totalCount: 0,
    search: '',
    category: '',
    equipment: '',
    target: '',
    repCountingOnly: false,
    initialized: false,
    debounceTimer: null,
  }
};

// MediaPipe 33 Keypoint Pose Connections
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Arms
  [11, 23], [12, 24], [23, 24],                      // Torso
  [23, 25], [24, 26], [25, 27], [26, 28],            // Legs
  [27, 29], [28, 30], [29, 31], [30, 32],            // Feet
  [15, 17], [15, 19], [15, 21], [16, 18], [16, 20], [16, 22] // Hands
];

// DOM Registry
const DOM = {
  // Session Stats
  sessionTimer: document.querySelector('#sessionTimer'),
  estCalories: document.querySelector('#estCalories'),
  aiStatusPill: document.querySelector('#aiStatusPill'),
  aiStatusText: document.querySelector('#aiStatusText'),
  resetWorkoutBtn: document.querySelector('#resetWorkoutBtn'),

  // Mode Tabs
  tabLiveMode: document.querySelector('#tabLiveMode'),
  tabUploadMode: document.querySelector('#tabUploadMode'),
  tabLibraryMode: document.querySelector('#tabLibraryMode'),
  liveArena: document.querySelector('#liveArena'),
  uploadArena: document.querySelector('#uploadArena'),
  libraryArena: document.querySelector('#libraryArena'),
  libraryCountBadge: document.querySelector('#libraryCountBadge'),

  // Exercise Library
  librarySearchInput: document.querySelector('#librarySearchInput'),
  filterCategory: document.querySelector('#filterCategory'),
  filterEquipment: document.querySelector('#filterEquipment'),
  filterTarget: document.querySelector('#filterTarget'),
  filterRepCounting: document.querySelector('#filterRepCounting'),
  libraryResultsCount: document.querySelector('#libraryResultsCount'),
  libraryPrevBtn: document.querySelector('#libraryPrevBtn'),
  libraryNextBtn: document.querySelector('#libraryNextBtn'),
  libraryPageInfo: document.querySelector('#libraryPageInfo'),
  libraryGrid: document.querySelector('#libraryGrid'),

  // Exercise Detail Modal
  exerciseDetailModal: document.querySelector('#exerciseDetailModal'),
  closeExerciseModal: document.querySelector('#closeExerciseModal'),
  modalExerciseGif: document.querySelector('#modalExerciseGif'),
  modalExerciseName: document.querySelector('#modalExerciseName'),
  modalCategory: document.querySelector('#modalCategory'),
  modalEquipment: document.querySelector('#modalEquipment'),
  modalTarget: document.querySelector('#modalTarget'),
  modalRepSupport: document.querySelector('#modalRepSupport'),
  modalTargetMuscle: document.querySelector('#modalTargetMuscle'),
  modalMuscleGroup: document.querySelector('#modalMuscleGroup'),
  modalSecondary: document.querySelector('#modalSecondary'),
  modalInstructions: document.querySelector('#modalInstructions'),

  // Rep Trackers
  cardCurl: document.querySelector('#cardCurl'),
  cardPress: document.querySelector('#cardPress'),
  cardSquat: document.querySelector('#cardSquat'),
  curlCount: document.querySelector('#curlCount'),
  pressCount: document.querySelector('#pressCount'),
  squatCount: document.querySelector('#squatCount'),
  curlStageBadge: document.querySelector('#curlStageBadge'),
  pressStageBadge: document.querySelector('#pressStageBadge'),
  squatStageBadge: document.querySelector('#squatStageBadge'),
  curlProgressBar: document.querySelector('#curlProgressBar'),
  pressProgressBar: document.querySelector('#pressProgressBar'),
  squatProgressBar: document.querySelector('#squatProgressBar'),

  // Viewport & Overlays
  webcamVideo: document.querySelector('#webcamVideo'),
  liveCanvas: document.querySelector('#liveCanvas'),
  cameraStartPrompt: document.querySelector('#cameraStartPrompt'),
  startCameraBtn: document.querySelector('#startCameraBtn'),
  stopCameraBtn: document.querySelector('#stopCameraBtn'),
  flipCameraBtn: document.querySelector('#flipCameraBtn'),
  goToUploadBtn: document.querySelector('#goToUploadBtn'),
  cameraErrorBox: document.querySelector('#cameraErrorBox'),
  cameraErrorMsg: document.querySelector('#cameraErrorMsg'),
  retryCamBtn: document.querySelector('#retryCamBtn'),
  useSampleBtn: document.querySelector('#useSampleBtn'),
  fpsCounter: document.querySelector('#fpsCounter'),
  actionBadgeHUD: document.querySelector('#actionBadgeHUD'),
  actionHUDText: document.querySelector('#actionHUDText'),
  stageBadgeHUD: document.querySelector('#stageBadgeHUD'),
  stageHUDText: document.querySelector('#stageHUDText'),
  repAnimationOverlay: document.querySelector('#repAnimationOverlay'),
  repPopExercise: document.querySelector('#repPopExercise'),

  // Toggles
  toggleSkeleton: document.querySelector('#toggleSkeleton'),
  toggleAngles: document.querySelector('#toggleAngles'),
  toggleAudio: document.querySelector('#toggleAudio'),

  // Upload Mode
  uploadDropArea: document.querySelector('#uploadDropArea'),
  filePicker: document.querySelector('#filePicker'),
  browseFileBtn: document.querySelector('#browseFileBtn'),
  uploadResultStage: document.querySelector('#uploadResultStage'),
  uploadPreviewImage: document.querySelector('#uploadPreviewImage'),
  uploadPreviewVideo: document.querySelector('#uploadPreviewVideo'),
  uploadCanvas: document.querySelector('#uploadCanvas'),
  changeFileBtn: document.querySelector('#changeFileBtn'),
  sampleSquatBtn: document.querySelector('#sampleSquatBtn'),
  samplePressBtn: document.querySelector('#samplePressBtn'),

  // Telemetry: Probability Meters
  probCurlLabel: document.querySelector('#probCurlLabel'),
  probCurlBar: document.querySelector('#probCurlBar'),
  probPressLabel: document.querySelector('#probPressLabel'),
  probPressBar: document.querySelector('#probPressBar'),
  probSquatLabel: document.querySelector('#probSquatLabel'),
  probSquatBar: document.querySelector('#probSquatBar'),

  // Telemetry: Kinematic Angles
  angleLeftElbow: document.querySelector('#angleLeftElbow'),
  angleRightElbow: document.querySelector('#angleRightElbow'),
  angleLeftKnee: document.querySelector('#angleLeftKnee'),
  angleRightKnee: document.querySelector('#angleRightKnee'),
  angleLeftHip: document.querySelector('#angleLeftHip'),
  angleRightHip: document.querySelector('#angleRightHip'),

  // AI Coach Note
  coachNoteText: document.querySelector('#coachNoteText'),

  // Sliders
  sliderDetect: document.querySelector('#sliderDetect'),
  valDetectText: document.querySelector('#valDetectText'),
  sliderTrack: document.querySelector('#sliderTrack'),
  valTrackText: document.querySelector('#valTrackText'),
  sliderClassify: document.querySelector('#sliderClassify'),
  valClassifyText: document.querySelector('#valClassifyText')
};

// Audio Rep Beep Context
let audioContext = null;

function playRepChime() {
  if (!state.audioEnabled) return;
  try {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
      audioContext.resume();
    }
    const osc = audioContext.createOscillator();
    const gain = audioContext.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5
    osc.frequency.exponentialRampToValueAtTime(783.99, audioContext.currentTime + 0.12); // G5
    gain.gain.setValueAtTime(0.28, audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.28);
    osc.connect(gain);
    gain.connect(audioContext.destination);
    osc.start();
    osc.stop(audioContext.currentTime + 0.28);
  } catch (e) {}
}

// App Initialization
document.addEventListener('DOMContentLoaded', () => {
  verifyHealth();
  attachEvents();
  startSessionClock();
});

// --- Session Clock & Calories ---
function startSessionClock() {
  if (state.sessionTimerId) clearInterval(state.sessionTimerId);
  state.sessionTimerId = setInterval(() => {
    if (state.isStreaming) {
      state.sessionSeconds++;
      const mins = String(Math.floor(state.sessionSeconds / 60)).padStart(2, '0');
      const secs = String(state.sessionSeconds % 60).padStart(2, '0');
      DOM.sessionTimer.textContent = `${mins}:${secs}`;
    }
  }, 1000);
}

function updateEstimatedCalories(counters) {
  const cals = (counters.curl * 0.32) + (counters.press * 0.42) + (counters.squat * 0.65) + (state.sessionSeconds * 0.05);
  DOM.estCalories.textContent = `${Math.round(cals)} kcal`;
}

// --- Health Check ---
async function verifyHealth() {
  try {
    const res = await fetch('/api/v1/health');
    const data = await res.json();
    if (res.ok && data.status === 'ok') {
      DOM.aiStatusText.textContent = 'Vision AI Active';
      DOM.aiStatusPill.style.color = 'var(--accent-emerald)';
      if (data.total_exercises && DOM.libraryCountBadge) {
        DOM.libraryCountBadge.textContent = Number(data.total_exercises).toLocaleString();
      }
    } else {
      DOM.aiStatusText.textContent = 'Degraded';
      DOM.aiStatusPill.style.color = 'var(--accent-amber)';
    }
  } catch (e) {
    DOM.aiStatusText.textContent = 'Offline';
    DOM.aiStatusPill.style.color = 'var(--accent-rose)';
  }
}

// --- Event Handlers ---
function attachEvents() {
  // Mode Switchers
  DOM.tabLiveMode.addEventListener('click', () => setMode('live'));
  DOM.tabUploadMode.addEventListener('click', () => setMode('upload'));
  if (DOM.tabLibraryMode) {
    DOM.tabLibraryMode.addEventListener('click', () => setMode('library'));
  }
  attachLibraryEvents();

  // Camera Actions
  DOM.startCameraBtn.addEventListener('click', startWebcam);
  DOM.retryCamBtn.addEventListener('click', startWebcam);
  DOM.stopCameraBtn.addEventListener('click', stopWebcam);
  DOM.flipCameraBtn.addEventListener('click', flipCameraLens);
  DOM.resetWorkoutBtn.addEventListener('click', resetAllWorkoutStats);

  DOM.goToUploadBtn.addEventListener('click', () => setMode('upload'));
  DOM.useSampleBtn.addEventListener('click', () => {
    setMode('upload');
    loadSample('/samples/sample_squat.webp');
  });

  // Sample Buttons
  DOM.sampleSquatBtn.addEventListener('click', () => loadSample('/samples/sample_squat.webp'));
  DOM.samplePressBtn.addEventListener('click', () => loadSample('/samples/sample_press.jpg'));

  // Audio Toggle
  DOM.toggleAudio.addEventListener('change', (e) => {
    state.audioEnabled = e.target.checked;
  });

  // Slider Updates
  [DOM.sliderDetect, DOM.sliderTrack, DOM.sliderClassify].forEach(slider => {
    slider.addEventListener('input', syncSliderSettings);
  });

  // File Upload
  DOM.browseFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    DOM.filePicker.click();
  });
  DOM.changeFileBtn.addEventListener('click', () => {
    DOM.uploadResultStage.hidden = true;
    DOM.uploadDropArea.hidden = false;
    DOM.filePicker.click();
  });
  DOM.uploadDropArea.addEventListener('click', () => DOM.filePicker.click());
  DOM.filePicker.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleUploadedMedia(e.target.files[0]);
    }
  });

  bindDragAndDrop();
}

// --- Mode Switching ---
function setMode(mode) {
  state.activeMode = mode;
  if (mode === 'live') {
    DOM.tabLiveMode?.classList.add('active');
    DOM.tabUploadMode?.classList.remove('active');
    DOM.tabLibraryMode?.classList.remove('active');
    if (DOM.liveArena) DOM.liveArena.hidden = false;
    if (DOM.uploadArena) DOM.uploadArena.hidden = true;
    if (DOM.libraryArena) DOM.libraryArena.hidden = true;
    stopVideoFilePlayback();
  } else if (mode === 'upload') {
    DOM.tabUploadMode?.classList.add('active');
    DOM.tabLiveMode?.classList.remove('active');
    DOM.tabLibraryMode?.classList.remove('active');
    if (DOM.uploadArena) DOM.uploadArena.hidden = false;
    if (DOM.liveArena) DOM.liveArena.hidden = true;
    if (DOM.libraryArena) DOM.libraryArena.hidden = true;
    stopWebcam();
  } else if (mode === 'library') {
    DOM.tabLibraryMode?.classList.add('active');
    DOM.tabLiveMode?.classList.remove('active');
    DOM.tabUploadMode?.classList.remove('active');
    if (DOM.libraryArena) DOM.libraryArena.hidden = false;
    if (DOM.liveArena) DOM.liveArena.hidden = true;
    if (DOM.uploadArena) DOM.uploadArena.hidden = true;
    stopWebcam();
    stopVideoFilePlayback();
    if (!state.library.initialized) {
      initExerciseLibrary();
    }
  }
}

// --- Robust WebRTC Camera Acquisition with Fallback Cascade ---
async function acquireCameraStream() {
  // Strategy 1: Ideal resolution + preferred facingMode
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: state.facingMode,
        width: { ideal: 640 },
        height: { ideal: 480 }
      },
      audio: false
    });
  } catch (e1) {
    console.warn('Strategy 1 camera constraints failed:', e1);
  }

  // Strategy 2: Basic video constraint (bypasses facingMode conflicts on Windows desktops)
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
  } catch (e2) {
    console.warn('Strategy 2 basic video failed:', e2);
  }

  // Strategy 3: Enumerate devices directly
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');
    if (videoInputs.length > 0) {
      return await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { exact: videoInputs[0].deviceId } },
        audio: false
      });
    }
  } catch (e3) {
    console.warn('Strategy 3 deviceId selection failed:', e3);
  }

  throw new Error('Camera device is busy, locked by another application, or denied by browser.');
}

async function startWebcam() {
  stopWebcam();
  DOM.cameraErrorBox.hidden = true;

  try {
    const stream = await acquireCameraStream();
    state.webcamStream = stream;
    DOM.webcamVideo.srcObject = stream;
    DOM.cameraStartPrompt.hidden = true;
    DOM.stopCameraBtn.disabled = false;
    state.isStreaming = true;

    DOM.webcamVideo.onloadedmetadata = () => {
      runStreamInference();
    };
  } catch (err) {
    console.error('Camera startup error:', err);
    DOM.cameraErrorMsg.textContent = `${err.message || 'Could not start camera source'}. Another program may be using your webcam, or camera permissions are blocked.`;
    DOM.cameraErrorBox.hidden = false;
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
  if (state.intervalTimer) {
    clearInterval(state.intervalTimer);
    state.intervalTimer = null;
  }
  DOM.webcamVideo.srcObject = null;
  DOM.cameraStartPrompt.hidden = false;
  DOM.stopCameraBtn.disabled = true;
  DOM.actionBadgeHUD.hidden = true;
  DOM.stageBadgeHUD.hidden = true;
  state.isStreaming = false;

  const ctx = DOM.liveCanvas.getContext('2d');
  ctx.clearRect(0, 0, DOM.liveCanvas.width, DOM.liveCanvas.height);
}

function flipCameraLens() {
  state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
  if (state.isStreaming) {
    startWebcam();
  }
}

// --- Live Webcam Inference Loop ---
function runStreamInference() {
  const offscreen = document.createElement('canvas');
  offscreen.width = 480;
  offscreen.height = 360;
  const offCtx = offscreen.getContext('2d');

  state.intervalTimer = setInterval(() => {
    if (!state.isStreaming || state.isProcessingFrame) return;

    const video = DOM.webcamVideo;
    if (!video.videoWidth || !video.videoHeight) return;

    state.isProcessingFrame = true;
    offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

    offscreen.toBlob(async (blob) => {
      if (!blob) {
        state.isProcessingFrame = false;
        return;
      }

      const form = new FormData();
      form.append('file', blob, 'frame.jpg');

      try {
        const response = await fetch('/api/v1/process_frame', {
          method: 'POST',
          body: form
        });

        if (response.ok) {
          const result = await response.json();
          if (result.status === 'success' && result.data) {
            refreshWorkoutDashboard(result.data, DOM.liveCanvas, video, true);
          }
        }
      } catch (e) {
      } finally {
        state.frameCount++;
        const now = performance.now();
        if (now - state.lastFpsTime >= 1000) {
          DOM.fpsCounter.textContent = Math.round((state.frameCount * 1000) / (now - state.lastFpsTime));
          state.frameCount = 0;
          state.lastFpsTime = now;
        }
        state.isProcessingFrame = false;
      }
    }, 'image/jpeg', 0.82);
  }, state.sendIntervalMs);
}

// --- Upload Media (Image or Video) Handling ---
function handleUploadedMedia(file) {
  if (!file) return;

  stopVideoFilePlayback();

  if (file.type.startsWith('video/')) {
    // Process as Workout Video
    processVideoFile(file);
  } else {
    // Process as Workout Image
    processImageFile(file);
  }
}

function processImageFile(file) {
  const url = URL.createObjectURL(file);
  DOM.uploadDropArea.hidden = true;
  DOM.uploadResultStage.hidden = false;
  DOM.uploadPreviewVideo.hidden = true;
  DOM.uploadPreviewImage.hidden = false;

  DOM.uploadPreviewImage.src = url;

  const formData = new FormData();
  formData.append('file', file);

  fetch('/api/v1/process_frame', {
    method: 'POST',
    body: formData
  })
  .then(res => res.json())
  .then(result => {
    if (result.status === 'success' && result.data) {
      const render = () => {
        refreshWorkoutDashboard(result.data, DOM.uploadCanvas, DOM.uploadPreviewImage, false);
      };
      if (DOM.uploadPreviewImage.complete) {
        render();
      } else {
        DOM.uploadPreviewImage.onload = render;
      }
    }
  })
  .catch(err => {
    alert('Error analyzing image: ' + err.message);
  });
}

function processVideoFile(file) {
  const url = URL.createObjectURL(file);
  DOM.uploadDropArea.hidden = true;
  DOM.uploadResultStage.hidden = false;
  DOM.uploadPreviewImage.hidden = true;
  DOM.uploadPreviewVideo.hidden = false;

  const video = DOM.uploadPreviewVideo;
  video.src = url;
  video.load();
  video.play().catch(() => {});

  const offscreen = document.createElement('canvas');
  offscreen.width = 480;
  offscreen.height = 360;
  const offCtx = offscreen.getContext('2d');

  state.videoPlaybackTimer = setInterval(() => {
    if (video.paused || video.ended || state.isProcessingFrame) return;

    if (!video.videoWidth || !video.videoHeight) return;

    state.isProcessingFrame = true;
    offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

    offscreen.toBlob(async (blob) => {
      if (!blob) {
        state.isProcessingFrame = false;
        return;
      }

      const form = new FormData();
      form.append('file', blob, 'video_frame.jpg');

      try {
        const response = await fetch('/api/v1/process_frame', {
          method: 'POST',
          body: form
        });

        if (response.ok) {
          const result = await response.json();
          if (result.status === 'success' && result.data) {
            refreshWorkoutDashboard(result.data, DOM.uploadCanvas, video, false);
          }
        }
      } catch (e) {
      } finally {
        state.isProcessingFrame = false;
      }
    }, 'image/jpeg', 0.82);
  }, 100);
}

function stopVideoFilePlayback() {
  if (state.videoPlaybackTimer) {
    clearInterval(state.videoPlaybackTimer);
    state.videoPlaybackTimer = null;
  }
  if (DOM.uploadPreviewVideo) {
    DOM.uploadPreviewVideo.pause();
    DOM.uploadPreviewVideo.src = '';
  }
}

function loadSample(samplePath) {
  fetch(samplePath)
    .then(r => r.blob())
    .then(blob => {
      const file = new File([blob], samplePath.split('/').pop(), { type: blob.type });
      handleUploadedMedia(file);
    })
    .catch(e => {
      alert('Could not load sample file: ' + e.message);
    });
}

// --- Dashboard & HUD Update ---
function refreshWorkoutDashboard(data, canvas, visualSource, isMirrored) {
  const counters = data.counters || { curl: 0, press: 0, squat: 0 };
  const stages = data.stages || { curl: null, press: null, squat: null };

  // 1. Update Rep Numbers & Goal Bars
  DOM.curlCount.textContent = counters.curl;
  DOM.pressCount.textContent = counters.press;
  DOM.squatCount.textContent = counters.squat;

  DOM.curlProgressBar.style.width = `${Math.min(100, (counters.curl / state.targets.curl) * 100)}%`;
  DOM.pressProgressBar.style.width = `${Math.min(100, (counters.press / state.targets.press) * 100)}%`;
  DOM.squatProgressBar.style.width = `${Math.min(100, (counters.squat / state.targets.squat) * 100)}%`;

  // 2. Stage Chips
  setPhaseChip(DOM.curlStageBadge, stages.curl);
  setPhaseChip(DOM.pressStageBadge, stages.press);
  setPhaseChip(DOM.squatStageBadge, stages.squat);

  // 3. Highlight Active Card
  DOM.cardCurl.classList.toggle('active', data.current_action === 'curl');
  DOM.cardPress.classList.toggle('active', data.current_action === 'press');
  DOM.cardSquat.classList.toggle('active', data.current_action === 'squat');

  // 4. Rep Completion Popup Event
  if (data.rep_event) {
    triggerRepModal(data.rep_event);
  }

  // 5. Update Estimated Calories
  updateEstimatedCalories(counters);

  // 6. On-Screen HUD Action & Stage Badges
  if (data.current_action) {
    const names = { curl: 'BICEP CURL', press: 'SHOULDER PRESS', squat: 'SQUAT' };
    const confPct = Math.round((data.confidence || 0) * 100);
    DOM.actionHUDText.textContent = `${names[data.current_action] || data.current_action.toUpperCase()} (${confPct}%)`;
    DOM.actionBadgeHUD.hidden = false;

    const activeStage = stages[data.current_action];
    if (activeStage) {
      DOM.stageHUDText.textContent = activeStage.toUpperCase();
      DOM.stageBadgeHUD.hidden = false;
    } else {
      DOM.stageBadgeHUD.hidden = true;
    }
  } else {
    DOM.actionBadgeHUD.hidden = true;
    DOM.stageBadgeHUD.hidden = true;
  }

  // 7. Probability Progress Bars
  const probs = data.probabilities || { curl: 0, press: 0, squat: 0 };
  const cPct = Math.round((probs.curl || 0) * 100);
  const pPct = Math.round((probs.press || 0) * 100);
  const sPct = Math.round((probs.squat || 0) * 100);

  DOM.probCurlLabel.textContent = `${cPct}%`;
  DOM.probCurlBar.style.width = `${cPct}%`;
  DOM.probPressLabel.textContent = `${pPct}%`;
  DOM.probPressBar.style.width = `${pPct}%`;
  DOM.probSquatLabel.textContent = `${sPct}%`;
  DOM.probSquatBar.style.width = `${sPct}%`;

  // 8. Kinematic Angles
  const angles = data.angles || {};
  DOM.angleLeftElbow.textContent = angles.left_elbow !== undefined ? `${angles.left_elbow}°` : '--°';
  DOM.angleRightElbow.textContent = angles.right_elbow !== undefined ? `${angles.right_elbow}°` : '--°';
  DOM.angleLeftKnee.textContent = angles.left_knee !== undefined ? `${angles.left_knee}°` : '--°';
  DOM.angleRightKnee.textContent = angles.right_knee !== undefined ? `${angles.right_knee}°` : '--°';
  DOM.angleLeftHip.textContent = angles.left_hip !== undefined ? `${angles.left_hip}°` : '--°';
  DOM.angleRightHip.textContent = angles.right_hip !== undefined ? `${angles.right_hip}°` : '--°';

  // 9. Dynamic AI Coaching Notes
  updateCoachTips(data.current_action, stages[data.current_action], angles);

  // 10. Draw Canvas Overlay
  drawSkeletonOverlay(canvas, visualSource, data.landmarks || [], angles, isMirrored);
}

function setPhaseChip(element, stage) {
  if (!stage) {
    element.textContent = 'REST';
    element.className = 'phase-chip';
  } else if (stage === 'up') {
    element.textContent = 'UP';
    element.className = 'phase-chip up';
  } else if (stage === 'down') {
    element.textContent = 'DOWN';
    element.className = 'phase-chip down';
  }
}

function triggerRepModal(exercise) {
  playRepChime();
  DOM.repPopExercise.textContent = `${exercise.toUpperCase()} COMPLETE!`;
  DOM.repAnimationOverlay.hidden = false;
  DOM.repAnimationOverlay.style.animation = 'none';
  DOM.repAnimationOverlay.offsetHeight; // trigger reflow
  DOM.repAnimationOverlay.style.animation = 'rep-pop 0.8s ease-out forwards';
  setTimeout(() => {
    DOM.repAnimationOverlay.hidden = true;
  }, 800);
}

function updateCoachTips(action, stage, angles) {
  if (!action) {
    DOM.coachNoteText.textContent = 'Keep your full body visible in frame to begin automatic tracking.';
    return;
  }
  if (action === 'curl') {
    if (stage === 'up') {
      DOM.coachNoteText.textContent = 'Peak contraction! Now lower arms slowly for maximum hypertrophy.';
    } else {
      DOM.coachNoteText.textContent = 'Keep upper arms locked against your torso while curling.';
    }
  } else if (action === 'press') {
    if (stage === 'up') {
      DOM.coachNoteText.textContent = 'Full extension overhead! Lower elbows down to shoulder height with control.';
    } else {
      DOM.coachNoteText.textContent = 'Drive the weight straight up without hyperextending your spine.';
    }
  } else if (action === 'squat') {
    if (stage === 'down') {
      DOM.coachNoteText.textContent = 'Great depth! Drive through heels and hips to stand tall.';
    } else {
      DOM.coachNoteText.textContent = 'Keep chest proud and track knees over toes as you sit back.';
    }
  }
}

// --- Canvas Skeleton & Joint Angle Visualization ---
function drawSkeletonOverlay(canvas, sourceEl, landmarks, angles, isMirrored) {
  if (!sourceEl || !canvas) return;

  const rect = sourceEl.getBoundingClientRect();
  const parentRect = sourceEl.parentElement.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  canvas.style.left = `${rect.left - parentRect.left}px`;
  canvas.style.top = `${rect.top - parentRect.top}px`;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (!landmarks || landmarks.length === 0) return;

  const lmMap = {};
  landmarks.forEach(lm => {
    lmMap[lm.id] = lm;
  });

  const natW = sourceEl.videoWidth || sourceEl.naturalWidth || rect.width;
  const natH = sourceEl.videoHeight || sourceEl.naturalHeight || rect.height;
  const scaleX = rect.width / natW;
  const scaleY = rect.height / natH;

  // 1. Draw Skeletal Connections
  if (DOM.toggleSkeleton.checked) {
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3.5;
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#10b981';

    POSE_CONNECTIONS.forEach(([i1, i2]) => {
      const p1 = lmMap[i1];
      const p2 = lmMap[i2];
      if (p1 && p2 && p1.visibility > 0.35 && p2.visibility > 0.35) {
        const x1 = isMirrored ? (natW - p1.x) * scaleX : p1.x * scaleX;
        const y1 = p1.y * scaleY;
        const x2 = isMirrored ? (natW - p2.x) * scaleX : p2.x * scaleX;
        const y2 = p2.y * scaleY;

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    });
  }

  // 2. Draw Landmark Joints
  landmarks.forEach(lm => {
    if (lm.visibility > 0.35) {
      const x = isMirrored ? (natW - lm.x) * scaleX : lm.x * scaleX;
      const y = lm.y * scaleY;

      ctx.beginPath();
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#06b6d4';
      ctx.shadowBlur = 6;
      ctx.shadowColor = '#06b6d4';
      ctx.fill();
    }
  });

  // 3. Draw Floating Kinematic Angle Labels
  if (DOM.toggleAngles.checked && angles) {
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillStyle = '#ffffff';
    ctx.shadowBlur = 6;
    ctx.shadowColor = '#000000';

    const drawTag = (id, val) => {
      const lm = lmMap[id];
      if (lm && lm.visibility > 0.35 && val !== undefined) {
        const x = isMirrored ? (natW - lm.x) * scaleX : lm.x * scaleX;
        const y = lm.y * scaleY;
        ctx.fillText(`${Math.round(val)}°`, x + 8, y + 4);
      }
    };

    drawTag(13, angles.left_elbow);
    drawTag(14, angles.right_elbow);
    drawTag(25, angles.left_knee);
    drawTag(26, angles.right_knee);
  }
}

// --- Reset Workout Stats ---
async function resetAllWorkoutStats() {
  try {
    const res = await fetch('/api/v1/reset_counters', { method: 'POST' });
    const data = await res.json();
    if (res.ok) {
      DOM.curlCount.textContent = '0';
      DOM.pressCount.textContent = '0';
      DOM.squatCount.textContent = '0';
      DOM.curlProgressBar.style.width = '0%';
      DOM.pressProgressBar.style.width = '0%';
      DOM.squatProgressBar.style.width = '0%';
      setPhaseChip(DOM.curlStageBadge, null);
      setPhaseChip(DOM.pressStageBadge, null);
      setPhaseChip(DOM.squatStageBadge, null);
      state.sessionSeconds = 0;
      DOM.sessionTimer.textContent = '00:00';
      DOM.estCalories.textContent = '0 kcal';
    }
  } catch (e) {}
}

// --- Sync Slider Settings ---
async function syncSliderSettings() {
  DOM.valDetectText.textContent = parseFloat(DOM.sliderDetect.value).toFixed(2);
  DOM.valTrackText.textContent = parseFloat(DOM.sliderTrack.value).toFixed(2);
  DOM.valClassifyText.textContent = parseFloat(DOM.sliderClassify.value).toFixed(2);

  try {
    await fetch('/api/v1/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        min_detection_confidence: parseFloat(DOM.sliderDetect.value),
        min_tracking_confidence: parseFloat(DOM.sliderTrack.value),
        classification_threshold: parseFloat(DOM.sliderClassify.value)
      })
    });
  } catch (e) {}
}

// --- Drag & Drop ---
function bindDragAndDrop() {
  const dropZone = DOM.uploadDropArea;
  if (!dropZone) return;

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
    window.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
    dropZone.addEventListener(evt, (e) => {
      e.preventDefault();
      e.stopPropagation();
    }, false);
  });

  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, () => dropZone.classList.add('dragging'), false);
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, () => dropZone.classList.remove('dragging'), false);
  });

  dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      handleUploadedMedia(file);
    }
  }, false);
}

// ═══════════════════════════════════════════════════════════════════════════
// EXERCISE LIBRARY CONTROLLER (1,324+ exercises, categories, GIF demonstrations)
// ═══════════════════════════════════════════════════════════════════════════

function attachLibraryEvents() {
  if (!DOM.libraryArena) return;

  // Search input with debounce
  DOM.librarySearchInput?.addEventListener('input', (e) => {
    clearTimeout(state.library.debounceTimer);
    state.library.debounceTimer = setTimeout(() => {
      state.library.search = e.target.value.trim();
      state.library.page = 1;
      fetchExercises();
    }, 300);
  });

  // Filter Dropdowns
  DOM.filterCategory?.addEventListener('change', (e) => {
    state.library.category = e.target.value;
    state.library.page = 1;
    fetchExercises();
  });

  DOM.filterEquipment?.addEventListener('change', (e) => {
    state.library.equipment = e.target.value;
    state.library.page = 1;
    fetchExercises();
  });

  DOM.filterTarget?.addEventListener('change', (e) => {
    state.library.target = e.target.value;
    state.library.page = 1;
    fetchExercises();
  });

  DOM.filterRepCounting?.addEventListener('change', (e) => {
    state.library.repCountingOnly = e.target.checked;
    state.library.page = 1;
    fetchExercises();
  });

  // Pagination Buttons
  DOM.libraryPrevBtn?.addEventListener('click', () => {
    if (state.library.page > 1) {
      state.library.page--;
      fetchExercises();
    }
  });

  DOM.libraryNextBtn?.addEventListener('click', () => {
    if (state.library.page < state.library.totalPages) {
      state.library.page++;
      fetchExercises();
    }
  });

  // Modal Close Events
  DOM.closeExerciseModal?.addEventListener('click', closeExerciseModal);
  DOM.exerciseDetailModal?.addEventListener('click', (e) => {
    if (e.target === DOM.exerciseDetailModal) {
      closeExerciseModal();
    }
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && DOM.exerciseDetailModal && !DOM.exerciseDetailModal.hidden) {
      closeExerciseModal();
    }
  });
}

async function initExerciseLibrary() {
  state.library.initialized = true;
  try {
    const res = await fetch('/api/v1/exercises/categories/summary');
    if (res.ok) {
      const data = await res.json();
      populateDropdown(DOM.filterCategory, data.categories, 'All Categories');
      populateDropdown(DOM.filterEquipment, data.equipment_types, 'All Equipment');
      populateDropdown(DOM.filterTarget, data.targets, 'All Muscles');
    }
  } catch (err) {
    console.warn('Failed to load category summaries:', err);
  }
  fetchExercises();
}

function populateDropdown(selectEl, dict, defaultLabel) {
  if (!selectEl || !dict) return;
  selectEl.innerHTML = `<option value="">${defaultLabel}</option>`;
  for (const [key, count] of Object.entries(dict)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = `${key} (${count})`;
    selectEl.appendChild(opt);
  }
}

async function fetchExercises() {
  if (!DOM.libraryGrid) return;
  DOM.libraryGrid.innerHTML = `
    <div style="grid-column: 1/-1; text-align: center; padding: 48px; color: var(--text-tertiary);">
      <div class="hud-spinner" style="display:inline-block; margin-bottom: 12px;"></div>
      <div>Loading exercises...</div>
    </div>
  `;

  const params = new URLSearchParams({
    page: String(state.library.page),
    per_page: String(state.library.perPage)
  });
  if (state.library.search) params.set('search', state.library.search);
  if (state.library.category) params.set('category', state.library.category);
  if (state.library.equipment) params.set('equipment', state.library.equipment);
  if (state.library.target) params.set('target', state.library.target);
  if (state.library.repCountingOnly) params.set('rep_counting', 'true');

  try {
    const res = await fetch(`/api/v1/exercises?${params.toString()}`);
    const data = await res.json();

    if (!res.ok || data.status !== 'success') {
      DOM.libraryGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 36px; color: var(--accent-rose);">Failed to load exercises.</div>`;
      return;
    }

    state.library.totalCount = data.total;
    state.library.totalPages = data.pages || 1;

    // Update results count & pagination info
    if (DOM.libraryResultsCount) {
      DOM.libraryResultsCount.textContent = `${Number(data.total).toLocaleString()} exercise${data.total === 1 ? '' : 's'} found`;
    }
    if (DOM.libraryPageInfo) {
      DOM.libraryPageInfo.textContent = `Page ${data.page} of ${Math.max(data.pages, 1)}`;
    }
    if (DOM.libraryPrevBtn) DOM.libraryPrevBtn.disabled = data.page <= 1;
    if (DOM.libraryNextBtn) DOM.libraryNextBtn.disabled = data.page >= data.pages;

    renderExerciseCards(data.exercises);
  } catch (err) {
    console.error('Error fetching exercises:', err);
    DOM.libraryGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 36px; color: var(--accent-rose);">Network error loading exercises.</div>`;
  }
}

function renderExerciseCards(exercises) {
  if (!DOM.libraryGrid) return;
  if (!exercises || exercises.length === 0) {
    DOM.libraryGrid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 48px; color: var(--text-tertiary);">
        <p style="font-size: 16px; font-weight: 600; color: var(--text-secondary); margin-bottom: 6px;">No exercises found</p>
        <p style="font-size: 13px;">Try clearing filters or adjusting your search term.</p>
      </div>
    `;
    return;
  }

  DOM.libraryGrid.innerHTML = exercises.map(ex => {
    const mediaPath = ex.image || ex.gif_url;
    const mediaUrl = mediaPath ? `/exercise-media/${mediaPath}` : '';
    const repBadge = ex.rep_counting_supported
      ? `<span class="card-chip rep-ok">AI Reps</span>`
      : '';
    const equipChip = ex.equipment ? `<span class="card-chip equipment">${escapeHtml(ex.equipment)}</span>` : '';
    const catChip = ex.category ? `<span class="card-chip">${escapeHtml(ex.category)}</span>` : '';

    return `
      <div class="exercise-lib-card" data-exercise-id="${escapeHtml(ex.id)}" tabindex="0" role="button">
        <img class="card-thumb" src="${escapeHtml(mediaUrl)}" alt="${escapeHtml(ex.name)}" loading="lazy" onerror="this.src='/samples/sample_squat.webp'; this.onerror=null;">
        <div class="card-label">
          <div class="card-name" title="${escapeHtml(ex.name)}">${escapeHtml(ex.name)}</div>
          <div class="card-meta">
            ${catChip}
            ${equipChip}
            ${repBadge}
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Attach card click handlers
  DOM.libraryGrid.querySelectorAll('.exercise-lib-card').forEach(card => {
    card.addEventListener('click', () => {
      const exId = card.getAttribute('data-exercise-id');
      if (exId) openExerciseModal(exId);
    });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        const exId = card.getAttribute('data-exercise-id');
        if (exId) openExerciseModal(exId);
      }
    });
  });
}

async function openExerciseModal(exerciseId) {
  if (!DOM.exerciseDetailModal) return;

  // Show modal immediately with loading state
  DOM.exerciseDetailModal.hidden = false;
  if (DOM.modalExerciseName) DOM.modalExerciseName.textContent = 'Loading...';
  if (DOM.modalExerciseGif) DOM.modalExerciseGif.src = '';
  if (DOM.modalCategory) DOM.modalCategory.textContent = '';
  if (DOM.modalEquipment) DOM.modalEquipment.textContent = '';
  if (DOM.modalTarget) DOM.modalTarget.textContent = '';
  if (DOM.modalRepSupport) DOM.modalRepSupport.textContent = '';
  if (DOM.modalTargetMuscle) DOM.modalTargetMuscle.textContent = '';
  if (DOM.modalMuscleGroup) DOM.modalMuscleGroup.textContent = '';
  if (DOM.modalSecondary) DOM.modalSecondary.textContent = '';
  if (DOM.modalInstructions) DOM.modalInstructions.textContent = 'Loading exercise details...';

  try {
    const res = await fetch(`/api/v1/exercises/${exerciseId}`);
    const data = await res.json();
    if (!res.ok || data.status !== 'success') {
      if (DOM.modalInstructions) DOM.modalInstructions.textContent = 'Failed to load details.';
      return;
    }

    const ex = data.exercise;
    if (DOM.modalExerciseName) DOM.modalExerciseName.textContent = ex.name || 'Exercise';

    // GIF media
    const gifPath = ex.gif_url || ex.image;
    if (DOM.modalExerciseGif) {
      DOM.modalExerciseGif.src = gifPath ? `/exercise-media/${gifPath}` : '';
      DOM.modalExerciseGif.alt = ex.name;
    }

    // Badges
    if (DOM.modalCategory) {
      DOM.modalCategory.textContent = ex.category || ex.body_part || 'General';
      DOM.modalCategory.hidden = !ex.category && !ex.body_part;
    }
    if (DOM.modalEquipment) {
      DOM.modalEquipment.textContent = ex.equipment || 'Bodyweight';
      DOM.modalEquipment.hidden = !ex.equipment;
    }
    if (DOM.modalTarget) {
      DOM.modalTarget.textContent = ex.target || '';
      DOM.modalTarget.hidden = !ex.target;
    }
    if (DOM.modalRepSupport) {
      const rep = ex.rep_counting || {};
      if (rep.supported) {
        DOM.modalRepSupport.textContent = 'AI Rep Counting Ready';
        DOM.modalRepSupport.className = 'detail-badge rep-support';
        DOM.modalRepSupport.hidden = false;
      } else {
        DOM.modalRepSupport.textContent = 'Manual Track';
        DOM.modalRepSupport.className = 'detail-badge';
        DOM.modalRepSupport.style.background = 'rgba(255,255,255,0.06)';
        DOM.modalRepSupport.style.color = 'var(--text-tertiary)';
        DOM.modalRepSupport.hidden = false;
      }
    }

    // Muscles
    if (DOM.modalTargetMuscle) DOM.modalTargetMuscle.textContent = ex.target || 'N/A';
    if (DOM.modalMuscleGroup) DOM.modalMuscleGroup.textContent = ex.body_part || ex.category || 'N/A';
    if (DOM.modalSecondary) {
      const sec = Array.isArray(ex.secondary_muscles) ? ex.secondary_muscles.join(', ') : (ex.secondary_muscles || 'None');
      DOM.modalSecondary.textContent = sec;
    }

    // Instructions
    let instText = '';
    if (ex.instructions) {
      if (typeof ex.instructions === 'object') {
        instText = ex.instructions.en || ex.instructions.es || Object.values(ex.instructions)[0] || '';
      } else {
        instText = String(ex.instructions);
      }
    }
    if (DOM.modalInstructions) {
      DOM.modalInstructions.textContent = instText || 'No instructions provided for this exercise.';
    }
  } catch (err) {
    console.error('Failed to load exercise details:', err);
    if (DOM.modalInstructions) DOM.modalInstructions.textContent = 'Failed to load exercise details.';
  }
}

function closeExerciseModal() {
  if (DOM.exerciseDetailModal) {
    DOM.exerciseDetailModal.hidden = true;
    if (DOM.modalExerciseGif) DOM.modalExerciseGif.src = '';
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
