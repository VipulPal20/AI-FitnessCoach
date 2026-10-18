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
  videoObjectUrl: null,
  // Frames carry real capture times (seconds) so the server's 1.9 s window
  // spans 1.9 s of actual movement regardless of how fast frames arrive.
  streamStartMs: 0,
  targetAction: null,

  // Workout Session State
  sessionSeconds: 0,
  sessionTimerId: null,
  totalReps: 0,
  targets: {
    curl: 12, press: 10,
    bicep_curl: 12, shoulder_press: 10, bench_press: 10,
    push_up: 12, squat: 15, deadlift: 10, pull_up: 8, row: 10,
    lunge: 12, jumping_jack: 20, unknown: 0,
  },

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
  },

  // Active Target Exercise from Library
  selectedExercise: null,
  activeExerciseDetail: null
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
  tabVideoMode: document.querySelector('#tabVideoMode'),
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
  btnStartWorkoutExercise: document.querySelector('#btnStartWorkoutExercise'),
  btnUploadExercise: document.querySelector('#btnUploadExercise'),

  // Active Exercise HUD & PiP Guide
  activeExerciseHUD: document.querySelector('#activeExerciseHUD'),
  activeExerciseName: document.querySelector('#activeExerciseName'),
  activeExerciseBadge: document.querySelector('#activeExerciseBadge'),
  toggleFormGuideBtn: document.querySelector('#toggleFormGuideBtn'),
  clearActiveExerciseBtn: document.querySelector('#clearActiveExerciseBtn'),
  formGuidePiP: document.querySelector('#formGuidePiP'),
  pipExerciseTitle: document.querySelector('#pipExerciseTitle'),
  pipExerciseGif: document.querySelector('#pipExerciseGif'),
  closeFormGuideBtn: document.querySelector('#closeFormGuideBtn'),

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

  // Studio Live Insights Cockpit (Right There with Screen)
  studioWorkspace: document.querySelector('.studio-workspace'),
  studioLiveInsights: document.querySelector('#studioLiveInsights'),
  studioActionName: document.querySelector('#studioActionName'),
  studioPhaseBadge: document.querySelector('#studioPhaseBadge'),
  studioBigRepCount: document.querySelector('#studioBigRepCount'),
  studioRepGoal: document.querySelector('#studioRepGoal'),
  studioProgressBar: document.querySelector('#studioProgressBar'),
  studioCoachNoteText: document.querySelector('#studioCoachNoteText'),
  coachFloatingHUD: document.querySelector('#coachFloatingHUD'),
  coachFloatingHUDText: document.querySelector('#coachFloatingHUDText'),
  studioAngleLeftElbow: document.querySelector('#studioAngleLeftElbow'),
  studioAngleRightElbow: document.querySelector('#studioAngleRightElbow'),
  studioAngleLeftKnee: document.querySelector('#studioAngleLeftKnee'),
  studioAngleRightKnee: document.querySelector('#studioAngleRightKnee'),
  studioAngleLeftHip: document.querySelector('#studioAngleLeftHip'),
  studioAngleRightHip: document.querySelector('#studioAngleRightHip'),
  studioProbRows: [1, 2, 3].map(rank => ({
    name: document.querySelector(`#studioProb${rank}Name`),
    bar: document.querySelector(`#studioProb${rank}Bar`),
    label: document.querySelector(`#studioProb${rank}Label`),
  })),
  repTargetSelect: document.querySelector('#repTargetSelect'),
  uploadAnalysisNote: document.querySelector('#uploadAnalysisNote'),
  sampleVideoBtn: document.querySelector('#sampleVideoBtn'),

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
  setRepTarget(null);
  attachEvents();
  startSessionClock();

  // URL Hash direct access: #library or #exercise-0001
  const hash = window.location.hash.toLowerCase();
  if (hash === '#library') {
    setMode('library');
  } else if (hash.startsWith('#exercise-')) {
    const exId = hash.replace('#exercise-', '');
    setMode('library');
    openExerciseModal(exId);
  }
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
  const count = (canonical, legacy) => Number(counters[canonical] ?? counters[legacy] ?? 0);
  const cals = (count('bicep_curl', 'curl') * 0.32)
    + ((count('shoulder_press', 'press') + count('bench_press')) * 0.42)
    + (count('squat') * 0.65)
    + (count('deadlift') * 0.75)
    + (count('jumping_jack') * 0.18)
    + (state.sessionSeconds * 0.05);
  DOM.estCalories.textContent = `${Math.round(cals)} kcal`;
}

// --- Health Check ---
async function verifyHealth() {
  try {
    const res = await fetch('/api/v1/health');
    const data = await res.json();
    if (res.ok && data.status === 'ok') {
      const experimental = data.experimental || data.model?.experimental || data.model?.production_ready === false;
      DOM.aiStatusText.textContent = experimental ? 'Experimental AI' : 'Vision AI Active';
      DOM.aiStatusPill.style.color = experimental ? 'var(--accent-amber)' : 'var(--accent-emerald)';
      if (data.total_exercises) {
        if (DOM.libraryCountBadge) DOM.libraryCountBadge.textContent = Number(data.total_exercises).toLocaleString();
        const navBadge = document.querySelector('#navLibraryCountBadge');
        if (navBadge) navBadge.textContent = Number(data.total_exercises).toLocaleString();
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
  DOM.tabVideoMode?.addEventListener('click', () => setMode('video'));
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
    loadSample('samples/sample_squat.webp');
  });

  // Sample Buttons
  DOM.sampleSquatBtn.addEventListener('click', () => loadSample('samples/sample_squat.webp'));
  DOM.samplePressBtn.addEventListener('click', () => loadSample('samples/sample_press.jpg'));
  DOM.sampleVideoBtn?.addEventListener('click', async () => {
    setMode('video');
    await setRepTarget('squat');
    loadSample('samples/sample_squat_video.webm');
  });

  // Rep target: count a chosen exercise, or follow the classifier
  DOM.repTargetSelect?.addEventListener('change', (e) => setRepTarget(e.target.value || null));

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
    DOM.tabVideoMode?.classList.remove('active');
    DOM.tabLibraryMode?.classList.remove('active');
    if (DOM.liveArena) DOM.liveArena.hidden = false;
    if (DOM.uploadArena) DOM.uploadArena.hidden = true;
    if (DOM.libraryArena) DOM.libraryArena.hidden = true;
    if (DOM.studioWorkspace) DOM.studioWorkspace.classList.remove('library-mode');
    if (DOM.studioLiveInsights) DOM.studioLiveInsights.hidden = false;
    stopVideoFilePlayback();
  } else if (mode === 'upload') {
    DOM.tabUploadMode?.classList.add('active');
    DOM.tabLiveMode?.classList.remove('active');
    DOM.tabVideoMode?.classList.remove('active');
    DOM.tabLibraryMode?.classList.remove('active');
    if (DOM.uploadArena) DOM.uploadArena.hidden = false;
    if (DOM.liveArena) DOM.liveArena.hidden = true;
    if (DOM.libraryArena) DOM.libraryArena.hidden = true;
    if (DOM.studioWorkspace) DOM.studioWorkspace.classList.remove('library-mode');
    if (DOM.studioLiveInsights) DOM.studioLiveInsights.hidden = false;
    if (DOM.filePicker) DOM.filePicker.accept = 'image/jpeg,image/png,image/webp,video/mp4,video/webm,video/ogg,video/quicktime';
    if (DOM.uploadResultStage) DOM.uploadResultStage.hidden = true;
    if (DOM.uploadDropArea) DOM.uploadDropArea.hidden = false;
    stopWebcam();
    stopVideoFilePlayback();
  } else if (mode === 'video') {
    DOM.tabVideoMode?.classList.add('active');
    DOM.tabLiveMode?.classList.remove('active');
    DOM.tabUploadMode?.classList.remove('active');
    DOM.tabLibraryMode?.classList.remove('active');
    if (DOM.uploadArena) DOM.uploadArena.hidden = false;
    if (DOM.liveArena) DOM.liveArena.hidden = true;
    if (DOM.libraryArena) DOM.libraryArena.hidden = true;
    if (DOM.studioWorkspace) DOM.studioWorkspace.classList.remove('library-mode');
    if (DOM.studioLiveInsights) DOM.studioLiveInsights.hidden = false;
    if (DOM.filePicker) DOM.filePicker.accept = 'video/mp4,video/webm,video/ogg,video/quicktime';
    if (DOM.uploadResultStage) DOM.uploadResultStage.hidden = true;
    if (DOM.uploadDropArea) DOM.uploadDropArea.hidden = false;
    stopWebcam();
    stopVideoFilePlayback();
  } else if (mode === 'library') {
    DOM.tabLibraryMode?.classList.add('active');
    DOM.tabLiveMode?.classList.remove('active');
    DOM.tabUploadMode?.classList.remove('active');
    DOM.tabVideoMode?.classList.remove('active');
    if (DOM.libraryArena) DOM.libraryArena.hidden = false;
    if (DOM.liveArena) DOM.liveArena.hidden = true;
    if (DOM.uploadArena) DOM.uploadArena.hidden = true;
    if (DOM.studioWorkspace) DOM.studioWorkspace.classList.add('library-mode');
    if (DOM.studioLiveInsights) DOM.studioLiveInsights.hidden = true;
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
    state.streamStartMs = performance.now();

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
    const capturedAt = (performance.now() - state.streamStartMs) / 1000;

    offscreen.toBlob(async (blob) => {
      if (!blob) {
        state.isProcessingFrame = false;
        return;
      }

      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      form.append('timestamp_seconds', capturedAt.toFixed(3));

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

  if (state.activeMode === 'video' && !file.type.startsWith('video/')) {
    alert('Video Analysis accepts video files only (MP4, WEBM, OGG, or MOV).');
    return;
  }

  stopVideoFilePlayback();

  if (file.type.startsWith('video/')) {
    // Process as Workout Video
    processVideoFile(file);
  } else {
    // Process as Workout Image
    processImageFile(file);
  }
}

function showUploadNote(text) {
  if (!DOM.uploadAnalysisNote) return;
  DOM.uploadAnalysisNote.textContent = text || '';
  DOM.uploadAnalysisNote.hidden = !text;
}

function processImageFile(file) {
  const url = URL.createObjectURL(file);
  showUploadNote('');
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
      const joints = (result.data.landmarks || []).filter(lm => lm.visibility > 0.35).length;
      showUploadNote(result.data.detected
        ? `Pose found: ${joints} of 33 joints visible, joint angles measured. Naming the exercise and counting reps need about 2 seconds of movement, so use Video Analysis or Live Camera for those.`
        : 'No person was detected in this photo. Try a photo with the whole body in view.');
    } else {
      showUploadNote(result.detail || 'The server could not analyze this image.');
    }
  })
  .catch(err => {
    alert('Error analyzing image: ' + err.message);
  });
}

async function processVideoFile(file) {
  const url = URL.createObjectURL(file);
  state.videoObjectUrl = url;
  DOM.uploadDropArea.hidden = true;
  DOM.uploadResultStage.hidden = false;
  DOM.uploadPreviewImage.hidden = true;
  DOM.uploadPreviewVideo.hidden = false;

  const video = DOM.uploadPreviewVideo;
  showUploadNote('');
  video.loop = false;
  video.src = url;
  video.load();
  video.onended = () => {
    const total = Object.values(state.lastCounters || {}).reduce((sum, n) => sum + Number(n || 0), 0);
    const parts = Object.entries(state.lastCounters || {})
      .filter(([, n]) => n > 0)
      .map(([action, n]) => `${n} × ${prettyAction(action)}`);
    showUploadNote(total
      ? `Video finished: ${parts.join(', ')} counted. Press play to watch again (counts carry on), or Reset to clear them.`
      : 'Video finished: no complete reps counted. Choose the exercise under "Count reps for", then press play again.');
  };
  // Each uploaded recording starts with a clean server-side sequence and
  // counters, so results belong to this video rather than the previous one.
  try {
    await fetch('/api/v1/reset_counters', { method: 'POST' });
  } catch (e) {
    // Continue playback if the reset endpoint is temporarily unavailable.
  }
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
    const capturedAt = video.currentTime;

    offscreen.toBlob(async (blob) => {
      if (!blob) {
        state.isProcessingFrame = false;
        return;
      }

      const form = new FormData();
      form.append('file', blob, 'video_frame.jpg');
      form.append('timestamp_seconds', capturedAt.toFixed(3));

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
  state.isProcessingFrame = false;
  if (DOM.uploadPreviewVideo) DOM.uploadPreviewVideo.onended = null;
  if (state.videoObjectUrl) {
    URL.revokeObjectURL(state.videoObjectUrl);
    state.videoObjectUrl = null;
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
const ACTION_LABELS = {
  bench_press: 'BENCH PRESS',
  shoulder_press: 'SHOULDER PRESS',
  push_up: 'PUSH-UP',
  squat: 'SQUAT',
  deadlift: 'DEADLIFT',
  bicep_curl: 'BICEP CURL',
  pull_up: 'PULL-UP',
  row: 'ROW',
  lunge: 'LUNGE',
  jumping_jack: 'JUMPING JACK',
  unknown: 'UNKNOWN',
  curl: 'BICEP CURL',
  press: 'SHOULDER PRESS',
};

function metricValue(metrics, canonical, legacy = canonical) {
  return Number(metrics?.[canonical] ?? metrics?.[legacy] ?? 0);
}

function prettyAction(action) {
  const label = ACTION_LABELS[action] || String(action || '').replace(/_/g, ' ').toUpperCase();
  return label.charAt(0) + label.slice(1).toLowerCase();
}

const STATUS_MESSAGES = {
  'Buffering pose window': 'Reading your movement…',
  'Stabilizing movement': 'Confirming the exercise…',
  'Low-confidence movement': 'Movement unclear: holding the last exercise.',
  'Pose gap; rebuilding window': 'Tracking interrupted: rebuilding.',
  'No visible pose': 'No person detected. Step back so your whole body is in frame.',
};

async function setRepTarget(action) {
  state.targetAction = action || null;
  if (DOM.repTargetSelect) DOM.repTargetSelect.value = state.targetAction || '';
  try {
    await fetch('/api/v1/target', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: state.targetAction }),
    });
  } catch (e) {}
}

function refreshWorkoutDashboard(data, canvas, visualSource, isMirrored) {
  const counters = data.counters || {};
  const stages = data.stages || {};
  const curlCount = metricValue(counters, 'bicep_curl', 'curl');
  const pressCount = metricValue(counters, 'shoulder_press', 'press');
  const squatCount = metricValue(counters, 'squat');

  // 1. Update Rep Numbers & Goal Bars (Section 2)
  DOM.curlCount.textContent = curlCount;
  DOM.pressCount.textContent = pressCount;
  DOM.squatCount.textContent = squatCount;

  DOM.curlProgressBar.style.width = `${Math.min(100, (curlCount / state.targets.bicep_curl) * 100)}%`;
  DOM.pressProgressBar.style.width = `${Math.min(100, (pressCount / state.targets.shoulder_press) * 100)}%`;
  DOM.squatProgressBar.style.width = `${Math.min(100, (squatCount / state.targets.squat) * 100)}%`;

  // 1b. Update Studio Live Insights Cockpit (Right There Alongside Screen!)
  state.lastCounters = counters;
  const currentAction = data.current_action;
  // Reps belong to the chosen target exercise if there is one, else the classifier's pick.
  const repAction = data.rep_action && data.rep_action !== 'unknown' ? data.rep_action : null;
  const currentRep = repAction ? Number(counters[repAction] || 0) : 0;
  const currentTarget = repAction && state.targets[repAction] ? state.targets[repAction] : 12;
  const currentStage = repAction ? stages[repAction] : null;

  if (DOM.studioActionName) {
    let title = 'READY TO TRAIN';
    if (repAction) {
      title = ACTION_LABELS[repAction] || repAction.toUpperCase();
      if (data.target_action && currentAction !== 'unknown' && currentAction !== data.target_action) {
        title += ` · AI SEES ${ACTION_LABELS[currentAction] || currentAction.toUpperCase()}`;
      }
    } else if (state.selectedExercise) {
      title = state.selectedExercise.name.toUpperCase();
    }
    DOM.studioActionName.textContent = title;
  }
  if (DOM.studioBigRepCount) {
    DOM.studioBigRepCount.textContent = currentRep;
  }
  if (DOM.studioRepGoal) {
    DOM.studioRepGoal.textContent = `Target: ${currentTarget}`;
  }
  if (DOM.studioProgressBar) {
    const pct = Math.min(100, Math.round((currentRep / currentTarget) * 100));
    DOM.studioProgressBar.style.width = `${pct}%`;
  }
  if (DOM.studioPhaseBadge) {
    setPhaseChip(DOM.studioPhaseBadge, currentStage);
  }

  // 2. Stage Chips
  setPhaseChip(DOM.curlStageBadge, stages.bicep_curl ?? stages.curl);
  setPhaseChip(DOM.pressStageBadge, stages.shoulder_press ?? stages.press);
  setPhaseChip(DOM.squatStageBadge, stages.squat);

  // 3. Highlight Active Card
  DOM.cardCurl.classList.toggle('active', data.current_action === 'bicep_curl' || data.current_action === 'curl');
  DOM.cardPress.classList.toggle('active', data.current_action === 'shoulder_press' || data.current_action === 'press');
  DOM.cardSquat.classList.toggle('active', data.current_action === 'squat');

  // 4. Rep Completion Popup Event
  if (data.rep_event) {
    triggerRepModal(data.rep_event);
  }

  // 5. Update Estimated Calories
  updateEstimatedCalories(counters);

  // 6. On-Screen HUD Action & Stage Badges
  if (repAction) {
    const confPct = Math.round((data.confidence || 0) * 100);
    DOM.actionHUDText.textContent = data.target_action
      ? `${ACTION_LABELS[repAction] || repAction.toUpperCase()} (TARGET)`
      : `${ACTION_LABELS[repAction] || repAction.toUpperCase()} (${confPct}%)`;
    DOM.actionBadgeHUD.hidden = false;

    if (currentStage) {
      DOM.stageHUDText.textContent = stageLabel(currentStage);
      DOM.stageBadgeHUD.hidden = false;
    } else {
      DOM.stageBadgeHUD.hidden = true;
    }
  } else {
    DOM.actionBadgeHUD.hidden = true;
    DOM.stageBadgeHUD.hidden = true;
  }

  // 7. Probability Progress Bars (Deep-dive + Studio Cockpit)
  const probs = data.probabilities || {};
  const cPct = Math.round(metricValue(probs, 'bicep_curl', 'curl') * 100);
  const pPct = Math.round(metricValue(probs, 'shoulder_press', 'press') * 100);
  const sPct = Math.round(metricValue(probs, 'squat') * 100);

  DOM.probCurlLabel.textContent = `${cPct}%`;
  DOM.probCurlBar.style.width = `${cPct}%`;
  DOM.probPressLabel.textContent = `${pPct}%`;
  DOM.probPressBar.style.width = `${pPct}%`;
  DOM.probSquatLabel.textContent = `${sPct}%`;
  DOM.probSquatBar.style.width = `${sPct}%`;

  // Studio cockpit shows the classifier's three most likely movements.
  const ranked = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 3);
  DOM.studioProbRows.forEach((row, index) => {
    const [action, prob] = ranked[index] || [null, 0];
    const pct = Math.round(Number(prob || 0) * 100);
    if (row.name) row.name.textContent = action && pct > 0 ? prettyAction(action) : '—';
    if (row.bar) row.bar.style.width = `${pct}%`;
    if (row.label) row.label.textContent = `${pct}%`;
  });

  // 8. Kinematic Angles (Deep-dive + Studio Cockpit)
  const angles = data.angles || {};
  const lElbow = angles.left_elbow !== undefined ? `${angles.left_elbow}°` : '--°';
  const rElbow = angles.right_elbow !== undefined ? `${angles.right_elbow}°` : '--°';
  const lKnee = angles.left_knee !== undefined ? `${angles.left_knee}°` : '--°';
  const rKnee = angles.right_knee !== undefined ? `${angles.right_knee}°` : '--°';
  const lHip = angles.left_hip !== undefined ? `${angles.left_hip}°` : '--°';
  const rHip = angles.right_hip !== undefined ? `${angles.right_hip}°` : '--°';

  DOM.angleLeftElbow.textContent = lElbow;
  DOM.angleRightElbow.textContent = rElbow;
  DOM.angleLeftKnee.textContent = lKnee;
  DOM.angleRightKnee.textContent = rKnee;
  DOM.angleLeftHip.textContent = lHip;
  DOM.angleRightHip.textContent = rHip;

  if (DOM.studioAngleLeftElbow) DOM.studioAngleLeftElbow.textContent = lElbow;
  if (DOM.studioAngleRightElbow) DOM.studioAngleRightElbow.textContent = rElbow;
  if (DOM.studioAngleLeftKnee) DOM.studioAngleLeftKnee.textContent = lKnee;
  if (DOM.studioAngleRightKnee) DOM.studioAngleRightKnee.textContent = rKnee;
  if (DOM.studioAngleLeftHip) DOM.studioAngleLeftHip.textContent = lHip;
  if (DOM.studioAngleRightHip) DOM.studioAngleRightHip.textContent = rHip;

  // 9. Dynamic AI Coaching Notes (Deep-dive + Studio Cockpit + On-screen floating HUD)
  updateCoachTips(repAction, currentStage, angles, data.message);

  // 10. Draw Canvas Overlay
  drawSkeletonOverlay(canvas, visualSource, data.landmarks || [], angles, isMirrored, data.image);
}

// The server reports 'home' (start position) and 'effort' (working position).
function stageLabel(stage) {
  if (stage === 'home' || stage === 'up') return 'START';
  if (stage === 'effort' || stage === 'down') return 'EFFORT';
  return 'REST';
}

function setPhaseChip(element, stage) {
  if (!element) return;
  element.textContent = stageLabel(stage);
  if (stage === 'home' || stage === 'up') {
    element.className = 'phase-chip up';
  } else if (stage === 'effort' || stage === 'down') {
    element.className = 'phase-chip down';
  } else {
    element.className = 'phase-chip';
  }
}

function triggerRepModal(exercise) {
  playRepChime();
  DOM.repPopExercise.textContent = `${ACTION_LABELS[exercise] || exercise.toUpperCase()} COMPLETE!`;
  DOM.repAnimationOverlay.hidden = false;
  DOM.repAnimationOverlay.style.animation = 'none';
  DOM.repAnimationOverlay.offsetHeight; // trigger reflow
  DOM.repAnimationOverlay.style.animation = 'rep-pop 0.8s ease-out forwards';
  setTimeout(() => {
    DOM.repAnimationOverlay.hidden = true;
  }, 800);
}

function updateCoachTips(action, stage, angles, message) {
  let cue = STATUS_MESSAGES[message] || message || 'Position your full body in frame to track movement & posture.';
  // Translate server stages into the cue vocabulary: a curl's effort is the top
  // of the movement; for presses and squats the effort is the bottom.
  if (stage === 'effort') stage = action === 'bicep_curl' ? 'up' : 'down';
  else if (stage === 'home') stage = action === 'bicep_curl' ? 'down' : 'up';
  if (action === 'curl' || action === 'bicep_curl') {
    if (stage === 'up') {
      cue = 'Peak contraction! Now lower arms slowly for maximum hypertrophy.';
    } else {
      cue = 'Keep upper arms locked against your torso while curling.';
    }
  } else if (action === 'press' || action === 'shoulder_press' || action === 'bench_press') {
    if (stage === 'up') {
      cue = 'Full overhead lockout! Lower elbows to shoulder level with control.';
    } else {
      cue = 'Drive weight straight up without hyperextending your spine.';
    }
  } else if (action === 'squat' || action === 'lunge') {
    if (stage === 'down') {
      cue = 'Great depth! Drive through heels and hips to stand tall.';
    } else {
      cue = 'Keep chest proud and track knees over toes as you sit back.';
    }
  } else if (action === 'deadlift' || action === 'row') {
    cue = 'Keep your spine neutral and move through a controlled hinge or pull.';
  } else if (action === 'push_up' || action === 'pull_up') {
    cue = 'Keep your trunk stable and use a smooth, controlled range of motion.';
  } else if (action === 'jumping_jack') {
    cue = 'Keep the rhythm steady and land softly under control.';
  }

  // Update deep-dive kinematics coach panel
  if (DOM.coachNoteText) DOM.coachNoteText.textContent = cue;

  // Update Studio live insights cockpit coach card (right beside screen)
  if (DOM.studioCoachNoteText) DOM.studioCoachNoteText.textContent = cue;

  // Update on-screen floating coach HUD banner (directly on the video screen!)
  if (DOM.coachFloatingHUDText) DOM.coachFloatingHUDText.textContent = cue;
  if (DOM.coachFloatingHUD) DOM.coachFloatingHUD.hidden = !action && !message;
}

// --- Canvas Skeleton & Joint Angle Visualization ---
function drawSkeletonOverlay(canvas, sourceEl, landmarks, angles, isMirrored, frameSize) {
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

  // Landmarks are pixels of the frame the server analysed (the 480x360 capture
  // canvas for webcam/video, the original file for photos), not of the element.
  const natW = frameSize?.width || sourceEl.videoWidth || sourceEl.naturalWidth || rect.width;
  const natH = frameSize?.height || sourceEl.videoHeight || sourceEl.naturalHeight || rect.height;
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
      state.lastCounters = {};
      DOM.sessionTimer.textContent = '00:00';
      DOM.estCalories.textContent = '0 kcal';
      if (DOM.studioBigRepCount) DOM.studioBigRepCount.textContent = '0';
      if (DOM.studioProgressBar) DOM.studioProgressBar.style.width = '0%';
      setPhaseChip(DOM.studioPhaseBadge, null);
      showUploadNote('');
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

  // Active Exercise HUD & PiP Controls
  DOM.clearActiveExerciseBtn?.addEventListener('click', () => {
    state.selectedExercise = null;
    setRepTarget(null);
    if (DOM.activeExerciseHUD) DOM.activeExerciseHUD.hidden = true;
    if (DOM.formGuidePiP) DOM.formGuidePiP.hidden = true;
    if (DOM.coachNoteText) {
      DOM.coachNoteText.textContent = 'AI Ready. Stand 6-8 feet away for full body detection.';
    }
  });

  DOM.toggleFormGuideBtn?.addEventListener('click', () => {
    if (DOM.formGuidePiP) {
      DOM.formGuidePiP.hidden = !DOM.formGuidePiP.hidden;
    }
  });

  DOM.closeFormGuideBtn?.addEventListener('click', () => {
    if (DOM.formGuidePiP) {
      DOM.formGuidePiP.hidden = true;
    }
  });

  // Top Nav Library Button
  const navLibBtn = document.querySelector('#navLibraryBtn');
  if (navLibBtn) {
    navLibBtn.addEventListener('click', () => {
      setMode('library');
      const studioSec = document.querySelector('#studio-section');
      if (studioSec) studioSec.scrollIntoView({ behavior: 'smooth' });
    });
  }

  // Smooth Scroll-Spy for Header Navigation Pills
  initScrollSpy();
}

function initScrollSpy() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.header-nav .nav-pill:not(.library-pill)');
  if (!sections.length || !navLinks.length) return;

  window.addEventListener('scroll', () => {
    let current = '';
    const scrollPos = window.scrollY + 140;

    sections.forEach(sec => {
      const top = sec.offsetTop;
      const height = sec.offsetHeight;
      if (scrollPos >= top && scrollPos < top + height) {
        current = sec.getAttribute('id');
      }
    });

    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${current}`) {
        link.classList.add('active');
      }
    });
  }, { passive: true });
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
    const repBadge = ex.rep_action
      ? `<span class="card-chip rep-ok" title="Counted as ${escapeHtml(prettyAction(ex.rep_action))}">AI Reps</span>`
      : '';
    const equipChip = ex.equipment ? `<span class="card-chip equipment">${escapeHtml(ex.equipment)}</span>` : '';
    const catChip = ex.category ? `<span class="card-chip">${escapeHtml(ex.category)}</span>` : '';

    return `
      <div class="exercise-lib-card" data-exercise-id="${escapeHtml(ex.id)}" tabindex="0" role="button">
        <button class="card-quick-train" data-quick-id="${escapeHtml(ex.id)}" type="button" title="Start training now with webcam">▶ Train</button>
        <img class="card-thumb" src="${escapeHtml(mediaUrl)}" alt="${escapeHtml(ex.name)}" loading="lazy" onerror="this.src='samples/sample_squat.webp'; this.onerror=null;">
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

  // Quick Train Button Handlers on each card
  DOM.libraryGrid.querySelectorAll('.card-quick-train').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const exId = btn.getAttribute('data-quick-id');
      const exObj = exercises.find(x => String(x.id) === String(exId));
      if (exObj) {
        startTrainingExercise(exObj, 'live');
      } else if (exId) {
        openExerciseModal(exId);
      }
    });
  });

  // Card click handlers (opens full detail modal)
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
    state.activeExerciseDetail = ex;

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

    // Wire up Direct Action Buttons
    if (DOM.btnStartWorkoutExercise) {
      DOM.btnStartWorkoutExercise.onclick = () => {
        startTrainingExercise(ex, 'live');
      };
    }
    if (DOM.btnUploadExercise) {
      DOM.btnUploadExercise.onclick = () => {
        startTrainingExercise(ex, 'video');
      };
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

// ── Direct Training Launcher for Any Library Exercise ───────────────────────
function startTrainingExercise(exercise, targetMode = 'live') {
  if (!exercise) return;
  state.selectedExercise = exercise;

  // 1. Update Active Exercise HUD in Live Arena
  if (DOM.activeExerciseHUD) {
    DOM.activeExerciseHUD.hidden = false;
  }
  if (DOM.activeExerciseName) {
    DOM.activeExerciseName.textContent = exercise.name || 'Exercise';
  }
  setRepTarget(exercise.rep_action || null);
  if (DOM.activeExerciseBadge) {
    const isRepReady = Boolean(exercise.rep_action);
    DOM.activeExerciseBadge.textContent = isRepReady ? `Counting ${prettyAction(exercise.rep_action)}` : 'Form Tracking';
    DOM.activeExerciseBadge.style.background = isRepReady ? 'rgba(16, 185, 129, 0.15)' : 'rgba(6, 182, 212, 0.15)';
    DOM.activeExerciseBadge.style.color = isRepReady ? 'var(--accent-emerald)' : 'var(--accent-cyan)';
    DOM.activeExerciseBadge.style.borderColor = isRepReady ? 'rgba(16, 185, 129, 0.3)' : 'rgba(6, 182, 212, 0.3)';
  }

  // 2. Picture-in-Picture floating form demonstration
  const mediaPath = exercise.gif_url || exercise.image;
  if (mediaPath && DOM.pipExerciseGif) {
    DOM.pipExerciseGif.src = `/exercise-media/${mediaPath}`;
    if (DOM.pipExerciseTitle) DOM.pipExerciseTitle.textContent = exercise.name;
    if (DOM.formGuidePiP) DOM.formGuidePiP.hidden = false;
  }

  // 3. Update AI Coach Note with exercise target & guidance
  let tip = '';
  if (exercise.instructions) {
    if (typeof exercise.instructions === 'object') {
      tip = exercise.instructions.en || Object.values(exercise.instructions)[0] || '';
    } else {
      tip = String(exercise.instructions);
    }
  }
  if (DOM.coachNoteText) {
    const targetInfo = exercise.target ? ` • Target: ${exercise.target}` : '';
    const firstSentence = tip ? tip.split('.')[0] + '.' : 'Maintain steady form throughout each repetition.';
    DOM.coachNoteText.textContent = `🎯 Training ${exercise.name}${targetInfo}. ${firstSentence}`;
  }

  // 4. Close the modal if open
  closeExerciseModal();

  // 5. Switch to desired mode (live or upload)
  setMode(targetMode);

  // 6. If live mode and webcam is not streaming, trigger start
  if (targetMode === 'live' && !state.isStreaming) {
    startWebcam();
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
