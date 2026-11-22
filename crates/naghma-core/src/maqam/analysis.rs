//! Rule-based maqam candidate scoring over a rolling note history.
//!
//! **The output is a heuristic ranking, not a probability.** Scores combine
//! configurable evidence terms; they say how well the recent notes fit each
//! maqam *as defined in the loaded dataset*, nothing more. With synthetic
//! test data the results are meaningless musically.
//!
//! ## Evidence considered
//!
//! For every maqam in the dataset and every candidate tonic:
//!
//! * **fit** — share of played time (duration-weighted) on notes that match a
//!   scale degree within the tolerance;
//! * **foreign notes** — share of played time on notes outside the scale
//!   (reported as contradicting evidence);
//! * **coverage** — share of the scale's degrees actually heard;
//! * **tonic** — emphasis on the tonic (time spent on it, and whether the
//!   most recent long note rests on it);
//! * **resting tones** — time spent on the dataset's resting steps;
//! * **ajnas** — completeness of the maqam's ajnas in what was heard;
//! * **phrases** — occurrences of the dataset's characteristic phrases in the
//!   melodic sequence;
//! * **cadence** — phrase endings (qaflat: long tones, tones before a pause,
//!   tremolo on one pitch) resting on the tonic;
//! * **register** — the melody lying from a little below the tonic to a
//!   little above its octave;
//! * **lower jins** — completeness of the jins on the tonic, which defines
//!   the maqam.
//!
//! Some maqamat are rotations of one another: Hijaz on Re (Hijaz on Re +
//! Nahawand on Sol) and Nahawand on Sol (Nahawand on Sol + Hijaz on Re) have
//! the same notes. Only cadence, register and what came before separate them.
//!
//! ## Two time scales
//!
//! A maqam lasts longer than [`AnalyzerConfig::window_s`]. Every score is
//! blended with a decaying history of that maqam and tonic
//! ([`ScoreWeights::memory`], [`AnalyzerConfig::memory_s`]); history fades
//! slowly while the notes stay inside the maqam's scale and quickly once they
//! leave it. A passage that dwells in an upper jins therefore keeps the qarar
//! established before it.
//!
//! The latest few seconds ([`AnalyzerConfig::jins_window_s`]) are read as a
//! jins: [`MaqamAnalysis::active_jins`] names the leading maqam's jins they
//! dwell in (its first jins, jins al-asl, or an upper one), and
//! [`MaqamAnalysis::foreign_jins`] a jins outside every documented route of
//! the maqam, a possible modulation.
//!
//! ## The performance as a whole
//!
//! A taqsim has one maqam it starts from and returns to, however many
//! passages wander elsewhere. [`MaqamAnalysis::performance`] ranks the
//! maqamat over everything played since the piece began (up to
//! [`AnalyzerConfig::performance_window_s`]), without any history blending:
//! the notes themselves are the history. It answers "which maqam is this
//! piece in", while [`MaqamAnalysis::candidates`] and the jins readings
//! answer "what is being played now". A silence longer than
//! [`AnalyzerConfig::performance_gap_s`] starts a new piece.
//!
//! Different maqamat can share most of their notes; when the best scores are
//! close the result is [`AnalysisStatus::Ambiguous`] rather than a forced
//! choice, and too little material gives
//! [`AnalysisStatus::InsufficientData`].

use super::{Jins, JinsRole, Maqam, MaqamDataset, ScaleDegree};
use crate::events::MusicalNoteEvent;
use crate::notation::SpelledNote;
use crate::tuning::NoteSpelling;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Which pitch identity the analysis uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PitchSpace {
    /// The player's notation (default; follows instrument transposition).
    #[default]
    Player,
    /// Sounding (concert) pitch.
    Sounding,
}

/// Weights of the evidence terms. All are configurable so the scoring can
/// be tuned and tested; none is claimed to be musically calibrated.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ScoreWeights {
    /// Weight of in-scale time share.
    pub fit: f64,
    /// Penalty per unit share of out-of-scale time.
    pub foreign: f64,
    /// Weight of scale-degree coverage.
    pub coverage: f64,
    /// Weight of time spent on the tonic (saturating at a 50 % share).
    pub tonic: f64,
    /// Weight of the latest long note resting on the tonic.
    pub final_rest: f64,
    /// Weight of resting-tone emphasis.
    pub resting: f64,
    /// Weight of jins completeness.
    pub ajnas: f64,
    /// Bonus per matched characteristic phrase (capped at 3).
    pub phrase: f64,
    /// Bonus for the maqam and tonic that led the previous analysis. A
    /// maqam lasts longer than one analysis step, so a near-tie keeps the
    /// current reading instead of flickering; clear evidence still wins.
    pub continuity: f64,
    /// Weight of phrase endings (qaflat) resting on the tonic. Phrases come
    /// to rest on the qarar; a dominant (ghammaz) is only a temporary stop.
    pub cadence: f64,
    /// Weight of the melody lying in the tonic's register: from a little
    /// below the tonic to a little above its octave. This is what separates
    /// rotations of one pitch set, such as Hijaz on Re and Nahawand on Sol:
    /// the same notes, but a different jins at the bottom.
    pub register: f64,
    /// Weight of the completeness of the lower jins (jins al-asl), which
    /// defines the maqam, in addition to [`Self::ajnas`].
    pub lower_jins: f64,
    /// Share of a candidate's score taken from its decaying score history
    /// (`0` = only the current window, `1` = only the history). Material
    /// that left the window still counts, fading over
    /// [`AnalyzerConfig::memory_s`], so a passage in an upper jins does not
    /// erase the qarar established before it.
    pub memory: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            fit: 1.0,
            foreign: 1.5,
            coverage: 0.5,
            tonic: 0.4,
            final_rest: 0.0,
            resting: 0.2,
            ajnas: 0.3,
            phrase: 0.15,
            continuity: 0.1,
            cadence: 0.2,
            register: 0.3,
            lower_jins: 0.2,
            memory: 0.5,
        }
    }
}

/// Parameters of [`MaqamAnalyzer`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AnalyzerConfig {
    /// Length of the rolling window scored for the maqam, in seconds.
    ///
    /// Short enough that a taqsim's sections are not mixed together: notes
    /// from a passage in another maqam leave the window quickly.
    pub window_s: f64,
    /// Seconds of playing used to estimate the instrument's tuning offset.
    /// Tuning belongs to the instrument, not the maqam, so it is estimated
    /// over a longer span than [`Self::window_s`] (never shorter).
    #[serde(default = "default_tuning_window")]
    pub tuning_window_s: f64,
    /// Minimum notes in the window before any ranking is reported.
    pub min_notes: usize,
    /// Minimum distinct pitch classes before any ranking is reported.
    pub min_distinct: usize,
    /// Accepted intonation error, in cents (1 cent = 1 % of a semitone, so
    /// 30 cents = 30 % of a semitone).
    ///
    /// Membership is soft: a played pitch within half of this of a scale
    /// degree counts fully as that degree, one exactly this far counts
    /// half, and credit fades smoothly to nothing at 1.5 × this distance.
    /// Real fingers and strings are never exact; the fade avoids an
    /// all-or-nothing cliff at the boundary.
    pub tolerance_cents: f64,
    /// Notes shorter than this (seconds) are ignored as transients.
    pub min_note_s: f64,
    /// Global tuning offset of the instrument in cents, subtracted from
    /// every note before scoring. `None` estimates it from the window
    /// (see [`estimate_tuning_offset`]); `Some(0.0)` disables compensation.
    pub tuning_offset_cents: Option<f64>,
    /// Largest offset the estimator may return, cents. Keep below 25: a
    /// larger shift could reinterpret naturals as quarter-tones.
    pub max_tuning_offset_cents: f64,
    /// Candidates whose score is within this margin of the best make the
    /// result ambiguous.
    pub ambiguity_margin: f64,
    /// Best scores below this give [`AnalysisStatus::Unknown`].
    pub min_score: f64,
    /// Maximum number of candidates returned.
    pub max_candidates: usize,
    /// Pitch identity to analyse.
    pub pitch_space: PitchSpace,
    /// Evidence weights.
    pub weights: ScoreWeights,
    /// Time constant of the score history, seconds (see
    /// [`ScoreWeights::memory`]).
    #[serde(default = "default_memory")]
    pub memory_s: f64,
    /// Time constant, seconds, with which the history follows a rising
    /// score once notes contradict the leading maqam: shorter than
    /// [`Self::memory_s`], so after a modulation the new maqam takes over
    /// quickly while, otherwise, an established one fades slowly.
    #[serde(default = "default_memory_rise")]
    pub memory_rise_s: f64,
    /// Length of the recent passage read for the active jins, seconds.
    /// Ajnas are short, so this is much shorter than [`Self::window_s`].
    #[serde(default = "default_jins_window")]
    pub jins_window_s: f64,
    /// Longest span, seconds, read as one performance for
    /// [`MaqamAnalysis::performance`]. `0` disables that reading (and keeps
    /// no notes beyond the other windows).
    #[serde(default = "default_performance_window")]
    pub performance_window_s: f64,
    /// A silence between notes at least this long, seconds, starts a new
    /// performance: what was played before it no longer counts toward
    /// [`MaqamAnalysis::performance`].
    #[serde(default = "default_performance_gap")]
    pub performance_gap_s: f64,
    /// Candidates of the performance reading within this margin of the best
    /// make it ambiguous. Larger than [`Self::ambiguity_margin`] would be
    /// too cautious here: over minutes, scores of maqamat that share notes
    /// converge, while their order is far more reliable than in a short
    /// window.
    #[serde(default = "default_performance_margin")]
    pub performance_ambiguity_margin: f64,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            memory_s: default_memory(),
            memory_rise_s: default_memory_rise(),
            jins_window_s: default_jins_window(),
            performance_window_s: default_performance_window(),
            performance_gap_s: default_performance_gap(),
            performance_ambiguity_margin: default_performance_margin(),
            window_s: 10.0,
            tuning_window_s: default_tuning_window(),
            min_notes: 6,
            min_distinct: 4,
            tolerance_cents: 30.0,
            min_note_s: 0.15,
            tuning_offset_cents: None,
            max_tuning_offset_cents: 20.0,
            ambiguity_margin: 0.08,
            min_score: 0.6,
            max_candidates: 5,
            pitch_space: PitchSpace::Player,
            weights: ScoreWeights::default(),
        }
    }
}

fn default_tuning_window() -> f64 {
    60.0
}

fn default_memory() -> f64 {
    60.0
}

fn default_memory_rise() -> f64 {
    10.0
}

fn default_jins_window() -> f64 {
    5.0
}

fn default_performance_window() -> f64 {
    300.0
}

fn default_performance_gap() -> f64 {
    20.0
}

fn default_performance_margin() -> f64 {
    0.08
}

/// One note in the analysis history.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HistoryNote {
    /// Start time, seconds (stream clock).
    pub start_s: f64,
    /// Duration, seconds.
    pub duration_s: f64,
    /// Pitch in semitones in the chosen pitch space (60 = Do4).
    pub pitch: f64,
}

/// Overall outcome of an analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnalysisStatus {
    /// Not enough notes or distinct pitches yet.
    InsufficientData,
    /// No dataset loaded, or no maqam fits well enough.
    Unknown,
    /// Several maqamat fit about equally well.
    Ambiguous,
    /// One candidate clearly fits best (still a heuristic judgement).
    Likely,
}

/// A piece of evidence for or against a candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum Evidence {
    /// Share of played time on scale notes.
    ScaleFit {
        /// Share in `[0, 1]`.
        share: f64,
    },
    /// Played notes not in the scale.
    ForeignNotes {
        /// Share of played time in `[0, 1]`.
        share: f64,
        /// The foreign pitches (semitones, rounded to quarter-tones).
        pitches: Vec<f64>,
    },
    /// Scale degrees heard / total.
    Coverage {
        /// Degrees heard.
        heard: usize,
        /// Degrees in the scale (excluding the octave).
        total: usize,
    },
    /// Emphasis on the tonic.
    Tonic {
        /// Share of played time on the tonic.
        share: f64,
        /// The latest note rests on the tonic.
        final_rest: bool,
    },
    /// Time on resting tones.
    RestingTones {
        /// Share of played time.
        share: f64,
    },
    /// Jins completeness.
    Jins {
        /// Jins id.
        jins: String,
        /// Fraction of its notes heard.
        completeness: f64,
    },
    /// Characteristic phrase found.
    Phrase {
        /// Index of the phrase in the maqam definition.
        index: usize,
    },
    /// Phrase endings (qaflat) resting on the tonic.
    Cadence {
        /// Share of cadence strength on the tonic, in `[0, 1]`.
        share: f64,
        /// Phrase endings found in the window.
        count: usize,
    },
    /// The melody lies in the tonic's register.
    Register {
        /// Share of played time, in `[0, 1]`.
        share: f64,
    },
}

/// A jins of the leading maqam that the most recent passage dwells in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveJins {
    /// Jins id in the dataset.
    pub jins: String,
    /// Scale step of the maqam the jins is placed on.
    pub on_step: u8,
    /// Its role in the maqam: the lower jins (jins al-asl) or an upper one
    /// (jins al-far', the second jins).
    pub role: JinsRole,
    /// Lowest note of the jins.
    pub root: NoteSpelling,
    /// Tonal centre of the jins (differs from `root` for ajnas such as
    /// Upper Rast, whose tonic is their top note).
    pub centre: NoteSpelling,
    /// Share of the passage's played time on the jins's notes.
    pub fit: f64,
}

/// A jins outside the leading maqam that explains the most recent passage
/// better than the maqam does: a possible modulation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectedJins {
    /// Jins id in the dataset.
    pub jins: String,
    /// Lowest note of the jins.
    pub root: NoteSpelling,
    /// Tonal centre of the jins.
    pub centre: NoteSpelling,
    /// Share of the passage's played time on the jins's notes.
    pub fit: f64,
}

/// A ranked maqam hypothesis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaqamCandidate {
    /// Maqam id in the dataset.
    pub maqam: String,
    /// Tonic assumed for this candidate.
    pub tonic: NoteSpelling,
    /// Selected alternative illustration; `None` means the primary formula.
    pub variant: Option<String>,
    /// Explicit anchor register for nonperiodic illustrations.
    pub tonic_octave: Option<i8>,
    /// Heuristic score (higher fits better; not a probability).
    pub score: f64,
    /// Supporting evidence.
    pub supporting: Vec<Evidence>,
    /// Contradicting evidence.
    pub contradicting: Vec<Evidence>,
}

/// The maqam of the performance as a whole: a ranking over everything
/// played since the piece began, rather than over the latest passage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceReading {
    /// Span read, `[start, end]` in stream seconds.
    pub window: (f64, f64),
    /// Notes in the span (after ignoring transients).
    pub notes: usize,
    /// Outcome ([`AnalysisStatus::InsufficientData`] is never reported:
    /// the reading is absent instead).
    pub status: AnalysisStatus,
    /// Ranked candidates (best first), one per maqam.
    pub candidates: Vec<MaqamCandidate>,
    /// Tuning offset (cents) subtracted from the notes before scoring.
    pub tuning_offset_cents: f64,
}

/// Result of [`MaqamAnalyzer::analyze`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaqamAnalysis {
    /// Analysed window `[start, end]` in stream seconds.
    pub window: (f64, f64),
    /// Notes in the window (after ignoring transients).
    pub notes: usize,
    /// Tonic fixed by the user, if any.
    pub tonic_selection: Option<NoteSpelling>,
    /// Overall status.
    pub status: AnalysisStatus,
    /// Ranked candidates (best first).
    pub candidates: Vec<MaqamCandidate>,
    /// The most recent passage leaves the leading maqam for a jins outside
    /// it ([`Self::foreign_jins`]): a modulation may be under way. Moving
    /// into the maqam's own upper jins is not a modulation.
    pub possible_modulation: bool,
    /// Tuning offset (cents) subtracted from the notes before scoring:
    /// estimated, or the configured fixed value.
    #[serde(default)]
    pub tuning_offset_cents: f64,
    /// Which jins of the leading maqam the most recent passage
    /// ([`AnalyzerConfig::jins_window_s`]) dwells in.
    #[serde(default)]
    pub active_jins: Option<ActiveJins>,
    /// A jins outside the leading maqam that explains the most recent
    /// passage better than the maqam does.
    #[serde(default)]
    pub foreign_jins: Option<DetectedJins>,
    /// The maqam of the whole performance so far, once enough of it has
    /// been heard. Unlike [`Self::candidates`] it does not follow a passage
    /// into another maqam, and it is still reported while the latest
    /// passage is too sparse to read ([`AnalysisStatus::InsufficientData`]).
    #[serde(default)]
    pub performance: Option<PerformanceReading>,
}

/// Decaying score of one maqam on one tonic pitch class.
#[derive(Debug, Clone, Copy)]
struct Remembered {
    tonic: NoteSpelling,
    score: f64,
}

/// Rolling note history plus scoring.
#[derive(Debug, Clone, Default)]
pub struct MaqamAnalyzer {
    config: AnalyzerConfig,
    history: VecDeque<HistoryNote>,
    /// Maqam id and tonic pitch class that led the previous analysis.
    leader: Option<(String, i64)>,
    /// Score history per maqam and tonic pitch class.
    memory: HashMap<(String, i64), Remembered>,
    /// Time of the last analysis that updated [`Self::memory`].
    remembered_at: Option<f64>,
    /// Foreign jins (id, root pitch class) of the previous analysis.
    pending_foreign: Option<(String, i64)>,
    /// Leader (maqam id, tonic pitch class) whose notes have left it for a
    /// confirmed foreign jins: its history no longer applies.
    modulating: Option<(String, i64)>,
    /// Until this time, rising evidence is followed quickly (see
    /// [`AnalyzerConfig::memory_rise_s`]).
    unsettled_until: f64,
}

impl MaqamAnalyzer {
    /// Create an analyser.
    pub fn new(config: AnalyzerConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            leader: None,
            memory: HashMap::new(),
            remembered_at: None,
            pending_foreign: None,
            modulating: None,
            unsettled_until: f64::NEG_INFINITY,
        }
    }

    /// Configuration.
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Notes currently in the history.
    pub fn history(&self) -> impl Iterator<Item = &HistoryNote> {
        self.history.iter()
    }

    /// Forget all notes.
    pub fn clear(&mut self) {
        self.history.clear();
        self.leader = None;
        self.memory.clear();
        self.remembered_at = None;
        self.pending_foreign = None;
        self.modulating = None;
        self.unsettled_until = f64::NEG_INFINITY;
    }

    /// Add a completed (or ongoing, with `now_s`) note event.
    pub fn push_event(&mut self, event: &MusicalNoteEvent, now_s: f64) {
        let pitch = match self.config.pitch_space {
            PitchSpace::Player => event.player_pitch.0,
            PitchSpace::Sounding => event.sounding_pitch.0,
        };
        self.push(HistoryNote {
            start_s: event.start_s,
            duration_s: event.duration_s(now_s).max(0.0),
            pitch,
        });
    }

    /// Add a note.
    pub fn push(&mut self, note: HistoryNote) {
        self.history.push_back(note);
    }

    /// Drop notes that ended before the maqam window, the tuning window and
    /// the performance window.
    pub fn prune(&mut self, now_s: f64) {
        let cutoff = now_s
            - self
                .config
                .window_s
                .max(self.config.tuning_window_s)
                .max(self.config.performance_window_s);
        while self
            .history
            .front()
            .is_some_and(|n| n.start_s + n.duration_s < cutoff)
        {
            self.history.pop_front();
        }
    }

    /// Score the current window against `dataset`, optionally with a fixed
    /// tonic.
    pub fn analyze(
        &mut self,
        dataset: Option<&MaqamDataset>,
        tonic: Option<NoteSpelling>,
        now_s: f64,
    ) -> MaqamAnalysis {
        self.prune(now_s);
        let config = &{ self.config };
        let kept: Vec<HistoryNote> = self
            .history
            .iter()
            .filter(|n| n.duration_s >= config.min_note_s)
            .copied()
            .collect();
        let performance = dataset
            .filter(|d| !d.maqamat.is_empty())
            .and_then(|d| read_performance(&kept, d, tonic, now_s, config));
        let tuning_from = now_s - config.window_s.max(config.tuning_window_s);
        let context: Vec<HistoryNote> = kept
            .iter()
            .filter(|n| n.start_s + n.duration_s >= tuning_from)
            .copied()
            .collect();
        let cutoff = now_s - config.window_s;
        let raw: Vec<HistoryNote> = context
            .iter()
            .filter(|n| n.start_s + n.duration_s >= cutoff)
            .copied()
            .collect();
        let window = (raw.first().map_or(now_s, |n| n.start_s), now_s);
        let offset = config.tuning_offset_cents.unwrap_or_else(|| {
            estimate_tuning_offset(&context, config.max_tuning_offset_cents).unwrap_or(0.0)
        });
        let notes = prepare(&raw, offset);
        let mut analysis = MaqamAnalysis {
            window,
            notes: notes.len(),
            tonic_selection: tonic,
            status: AnalysisStatus::InsufficientData,
            candidates: Vec::new(),
            possible_modulation: false,
            tuning_offset_cents: offset,
            active_jins: None,
            foreign_jins: None,
            performance,
        };
        let distinct = distinct_classes(&notes);
        if notes.len() < config.min_notes || distinct < config.min_distinct {
            self.leader = None;
            return analysis;
        }
        let Some(dataset) = dataset.filter(|d| !d.maqamat.is_empty()) else {
            self.leader = None;
            analysis.status = AnalysisStatus::Unknown;
            return analysis;
        };

        let tonics = match tonic {
            Some(t) => vec![t],
            None => {
                // The qarar of the current reading stays a candidate even
                // while a passage in an upper jins never touches it.
                let mut tonics = candidate_tonics(&notes);
                let mut remembered: Vec<&Remembered> = self.memory.values().collect();
                remembered.sort_by(|a, b| b.score.total_cmp(&a.score));
                for r in remembered.into_iter().take(3) {
                    let class = r.tonic.quarter_tone_class();
                    if !tonics.iter().any(|t| t.quarter_tone_class() == class) {
                        tonics.push(r.tonic);
                    }
                }
                tonics
            }
        };
        let mut candidates = rank_all(&notes, now_s, dataset, &tonics, config);
        self.remember(&mut candidates, now_s);
        dedup_by_maqam(&mut candidates);
        if let Some((maqam, class)) = &self.leader {
            if let Some(c) = candidates
                .iter_mut()
                .find(|c| &c.maqam == maqam && c.tonic.quarter_tone_class() == *class)
            {
                c.score += config.weights.continuity;
                candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            }
        }
        self.leader = candidates
            .first()
            .map(|c| (c.maqam.clone(), c.tonic.quarter_tone_class()));
        let best = candidates.first().map_or(0.0, |c| c.score);
        analysis.status = if best < config.min_score {
            AnalysisStatus::Unknown
        } else if candidates
            .get(1)
            .is_some_and(|c| best - c.score <= config.ambiguity_margin)
        {
            AnalysisStatus::Ambiguous
        } else {
            AnalysisStatus::Likely
        };

        // Which jins of the leader the latest passage dwells in, or which
        // jins outside it.
        if let Some(leader) = candidates.first() {
            let recent_from = now_s - config.jins_window_s;
            let recent: Vec<Prepared> = notes
                .iter()
                .filter(|n| n.start + n.duration >= recent_from)
                .copied()
                .collect();
            let (active, mut foreign) = read_jins(&recent, dataset, leader, config);
            // Only a jins heard in two analyses in a row: one misread note
            // is not a modulation.
            let key = foreign
                .as_ref()
                .map(|f| (f.jins.clone(), f.root.quarter_tone_class()));
            if key.is_none() || key != self.pending_foreign {
                foreign = None;
            }
            self.pending_foreign = key;
            analysis.possible_modulation = foreign.is_some();
            self.modulating = foreign
                .as_ref()
                .map(|_| (leader.maqam.clone(), leader.tonic.quarter_tone_class()));
            analysis.active_jins = active;
            analysis.foreign_jins = foreign;
        }

        candidates.truncate(config.max_candidates);
        analysis.candidates = candidates;
        analysis
    }

    /// Blend every candidate's score with the decaying history of its maqam
    /// and tonic pitch class, then update that history.
    fn remember(&mut self, candidates: &mut [MaqamCandidate], now_s: f64) {
        let beta = self.config.weights.memory.clamp(0.0, 1.0);
        let tau = self.config.memory_s;
        if beta <= 0.0 || tau <= 0.0 || candidates.is_empty() {
            self.memory.clear();
            self.remembered_at = None;
            return;
        }
        let key = |c: &MaqamCandidate| (c.maqam.clone(), c.tonic.quarter_tone_class());
        // Each maqam and tonic counts with its best alternative / register,
        // and with that candidate's share of notes outside its scale.
        let mut raw: HashMap<(String, i64), (Remembered, f64)> = HashMap::new();
        for c in candidates.iter() {
            let e = raw.entry(key(c)).or_insert((
                Remembered {
                    tonic: c.tonic,
                    score: f64::NEG_INFINITY,
                },
                1.0,
            ));
            if c.score > e.0.score {
                e.0.score = c.score;
                e.1 = foreign_share(c);
            }
        }
        // Absent and new hypotheses count as the weakest one present.
        let floor = raw
            .values()
            .map(|r| r.0.score)
            .fold(f64::INFINITY, f64::min);
        let dt = self.remembered_at.map(|t| (now_s - t).max(0.0));
        // A hypothesis the current notes are consistent with keeps its
        // history: resting on the ghammaz of an established maqam does not
        // undo it. Notes outside its scale make the history fade faster.
        // Once the notes contradict the leading maqam, rising evidence for
        // others is followed faster (`memory_rise_s`): the music has moved
        // and the new maqam takes over without a long wait. While the leader
        // still explains every note (an upper jins on its ghammaz has the
        // same notes as a maqam on that ghammaz), nothing is hurried.
        // The reading stays unsettled for a while after a contradiction, as
        // the lead may pass through hypotheses on the way to the new maqam.
        const CONTRADICTED: f64 = 0.1;
        let leader_contradicted = self
            .leader
            .as_ref()
            .and_then(|k| raw.get(k))
            .is_some_and(|r| r.1 >= CONTRADICTED);
        if leader_contradicted {
            self.unsettled_until = now_s + 2.0 * self.config.memory_rise_s;
        }
        let rise = if now_s <= self.unsettled_until {
            self.config.memory_rise_s.max(1e-3)
        } else {
            tau
        };
        let alpha = |foreign: f64, rising: bool| {
            dt.map_or(0.0, |dt| {
                let tau = if rising {
                    rise.min(tau)
                } else {
                    tau / (1.0 + 10.0 * foreign)
                };
                (-dt / tau).exp()
            })
        };
        let mut next = HashMap::new();
        for (k, old) in &self.memory {
            let (now, mut a) = raw.get(k).map_or((floor, alpha(0.0, false)), |r| {
                (r.0.score, alpha(r.1, r.0.score > old.score))
            });
            // A confirmed move to a jins outside the maqam: what established
            // the maqam no longer holds it in place.
            if self.modulating.as_ref() == Some(k) {
                a = 0.0;
            }
            next.insert(
                k.clone(),
                Remembered {
                    tonic: old.tonic,
                    score: a * old.score + (1.0 - a) * now,
                },
            );
        }
        for (k, (r, foreign)) in raw {
            let a = alpha(foreign, r.score > floor);
            next.entry(k).or_insert(Remembered {
                tonic: r.tonic,
                score: a * floor + (1.0 - a) * r.score,
            });
        }
        // Forget hypotheses that have faded into the background. Many share
        // the floor score; ties are broken by key, so which ones are kept
        // does not depend on the hash map's random iteration order.
        let mut kept: Vec<_> = next.into_iter().collect();
        kept.sort_by(|a, b| b.1.score.total_cmp(&a.1.score).then_with(|| a.0.cmp(&b.0)));
        kept.truncate(64);
        self.memory = kept.into_iter().collect();
        self.remembered_at = Some(now_s);
        for c in candidates.iter_mut() {
            if let Some(r) = self.memory.get(&key(c)) {
                c.score = (1.0 - beta) * c.score + beta * r.score;
            }
        }
        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    }
}

/// The notes of the current performance: those within
/// [`AnalyzerConfig::performance_window_s`] of `now_s` and after the latest
/// silence of at least [`AnalyzerConfig::performance_gap_s`].
fn performance_notes(
    notes: &[HistoryNote],
    now_s: f64,
    config: &AnalyzerConfig,
) -> Vec<HistoryNote> {
    let from = now_s - config.performance_window_s;
    let mut out: Vec<HistoryNote> = notes
        .iter()
        .filter(|n| n.start_s + n.duration_s >= from)
        .copied()
        .collect();
    out.sort_by(|a, b| a.start_s.total_cmp(&b.start_s));
    if config.performance_gap_s > 0.0 {
        let mut end = f64::NEG_INFINITY;
        let mut first = 0;
        for (i, n) in out.iter().enumerate() {
            if i > 0 && n.start_s - end >= config.performance_gap_s {
                first = i;
            }
            end = end.max(n.start_s + n.duration_s);
        }
        out.drain(..first);
    }
    out
}

/// Rank the maqamat over the whole performance (see
/// [`MaqamAnalysis::performance`]). `None` while disabled or too little has
/// been played.
fn read_performance(
    notes: &[HistoryNote],
    dataset: &MaqamDataset,
    tonic: Option<NoteSpelling>,
    now_s: f64,
    config: &AnalyzerConfig,
) -> Option<PerformanceReading> {
    if config.performance_window_s <= 0.0 {
        return None;
    }
    let raw = performance_notes(notes, now_s, config);
    let offset = config.tuning_offset_cents.unwrap_or_else(|| {
        estimate_tuning_offset(&raw, config.max_tuning_offset_cents).unwrap_or(0.0)
    });
    let prepared = prepare(&raw, offset);
    if prepared.len() < config.min_notes || distinct_classes(&prepared) < config.min_distinct {
        return None;
    }
    let tonics = tonic.map_or_else(|| candidate_tonics(&prepared), |t| vec![t]);
    let mut candidates = rank_all(&prepared, now_s, dataset, &tonics, config);
    dedup_by_maqam(&mut candidates);
    let best = candidates.first().map_or(0.0, |c| c.score);
    let status = if best < config.min_score {
        AnalysisStatus::Unknown
    } else if candidates
        .get(1)
        .is_some_and(|c| best - c.score <= config.performance_ambiguity_margin)
    {
        AnalysisStatus::Ambiguous
    } else {
        AnalysisStatus::Likely
    };
    candidates.truncate(config.max_candidates);
    Some(PerformanceReading {
        window: (prepared.first().map_or(now_s, |n| n.start), now_s),
        notes: prepared.len(),
        status,
        candidates,
        tuning_offset_cents: offset,
    })
}

/// Share of a candidate's played time outside its scale.
fn foreign_share(c: &MaqamCandidate) -> f64 {
    c.contradicting
        .iter()
        .find_map(|e| match e {
            Evidence::ForeignNotes { share, .. } => Some(*share),
            _ => None,
        })
        .unwrap_or(0.0)
}

/// Estimate an instrument's global tuning offset, in cents, from played
/// notes.
///
/// Arabic music uses quarter-tones, but most played time is on
/// 12-EDO-natural degrees, so the offset is the shift of the semitone grid
/// that best fits the notes: the maximum of a duration-weighted kernel
/// density (σ = 10 cents) of note positions relative to the nearest
/// semitone, searched within `±max_cents` and refined by a local weighted
/// mean. Notes far from any semitone (such as half-flats) barely contribute.
///
/// Returns `None` with less than one second of material. `max_cents` should
/// stay below 25: a 50-cent shift would turn every natural into a
/// quarter-tone.
pub fn estimate_tuning_offset(notes: &[HistoryNote], max_cents: f64) -> Option<f64> {
    const SIGMA: f64 = 10.0;
    let max_cents = max_cents.clamp(0.0, 49.0);
    let points: Vec<(f64, f64)> = notes
        .iter()
        .filter(|n| n.duration_s > 0.0 && n.pitch.is_finite())
        .map(|n| (n.pitch * 100.0, n.duration_s.min(2.0)))
        .collect();
    let total: f64 = points.iter().map(|p| p.1).sum();
    if total < 1.0 {
        return None;
    }
    let wrap = |x: f64| (x + 50.0).rem_euclid(100.0) - 50.0;
    let density = |d: f64| -> f64 {
        points
            .iter()
            .map(|&(c, w)| w * (-0.5 * (wrap(c - d) / SIGMA).powi(2)).exp())
            .sum()
    };
    let steps = (max_cents * 2.0).round() as i64;
    let mut best: (f64, f64) = (0.0, f64::NEG_INFINITY);
    for k in -steps..=steps {
        let d = k as f64 * 0.5;
        let s = density(d);
        // Prefer the smaller offset on (near-)ties.
        if s > best.1 + 1e-9 || ((s - best.1).abs() <= 1e-9 && d.abs() < best.0.abs()) {
            best = (d, s);
        }
    }
    let (mut num, mut den) = (0.0, 0.0);
    for &(c, w) in &points {
        let r = wrap(c - best.0);
        if r.abs() < 2.0 * SIGMA {
            num += w * r;
            den += w;
        }
    }
    let refined = if den > 0.0 {
        best.0 + num / den
    } else {
        best.0
    };
    Some(refined.clamp(-max_cents, max_cents))
}

/// A note prepared for scoring.
#[derive(Debug, Clone, Copy)]
struct Prepared {
    /// Absolute pitch in cents after tuning compensation.
    cents: f64,
    /// Evidence weight (duration, at least 50 ms).
    weight: f64,
    /// Start, seconds.
    start: f64,
    /// Duration, seconds.
    duration: f64,
}

fn prepare(notes: &[HistoryNote], offset: f64) -> Vec<Prepared> {
    let mut prepared: Vec<Prepared> = notes
        .iter()
        .map(|n| Prepared {
            cents: n.pitch * 100.0 - offset,
            weight: n.duration_s.max(0.05),
            start: n.start_s,
            duration: n.duration_s,
        })
        .collect();
    prepared.sort_by(|a, b| a.start.total_cmp(&b.start));
    prepared
}

/// A phrase ending: a tone the melody comes to rest on.
#[derive(Debug, Clone, Copy)]
struct Cadence {
    /// Absolute pitch in cents.
    cents: f64,
    /// Relative strength: longer, approached from above, followed by a pause.
    strength: f64,
}

/// Find phrase endings (qaflat) in time-ordered notes.
///
/// Repeated plucks of one pitch with no gap (a tremolo, rish) are one
/// sustained tone. A tone ends a phrase when a pause follows it or when it
/// lasts clearly longer than the passage's typical tone. Strength grows with
/// its length, with a falling approach (phrases descend to the qarar) and
/// with a following pause. `now_s` closes the last tone's pause.
fn cadences(notes: &[Prepared], now_s: f64) -> Vec<Cadence> {
    /// Tremolo repeats stay within this many cents.
    const SAME_CENTS: f64 = 35.0;
    /// ... and this gap, seconds.
    const JOIN_GAP_S: f64 = 0.12;
    /// A silence this long ends a phrase.
    const PAUSE_S: f64 = 0.3;
    struct Tone {
        cents: f64,
        weight: f64,
        start: f64,
        end: f64,
    }
    let mut tones: Vec<Tone> = Vec::new();
    for n in notes {
        let end = n.start + n.duration;
        if let Some(t) = tones.last_mut() {
            if (n.cents - t.cents).abs() <= SAME_CENTS && n.start - t.end <= JOIN_GAP_S {
                t.cents = (t.cents * t.weight + n.cents * n.weight) / (t.weight + n.weight);
                t.weight += n.weight;
                t.end = t.end.max(end);
                continue;
            }
        }
        tones.push(Tone {
            cents: n.cents,
            weight: n.weight,
            start: n.start,
            end,
        });
    }
    if tones.len() < 2 {
        return Vec::new();
    }
    let mut lengths: Vec<f64> = tones.iter().map(|t| t.end - t.start).collect();
    lengths.sort_by(f64::total_cmp);
    let long = (1.75 * lengths[lengths.len() / 2]).max(0.5);
    let mut out = Vec::new();
    for (i, t) in tones.iter().enumerate() {
        let length = t.end - t.start;
        let pause = tones.get(i + 1).map_or(now_s, |n| n.start) - t.end;
        let paused = pause >= PAUSE_S;
        if !paused && length < long {
            continue;
        }
        let falling = i > 0 && tones[i - 1].cents > t.cents + 50.0;
        out.push(Cadence {
            cents: t.cents,
            strength: length.clamp(0.1, 2.5)
                * if falling { 1.5 } else { 1.0 }
                * if paused { 1.5 } else { 1.0 },
        });
    }
    out
}

/// Credit for a tone `r` cents above the tonic lying in the tonic's
/// register: full from a little below the tonic (a leading tone or a third
/// below) to a little above its octave, fading to nothing at a fifth below
/// and at a fifth above the octave.
fn register_credit(r: f64) -> f64 {
    const LOW_ZERO: f64 = -750.0;
    const LOW_FULL: f64 = -250.0;
    const HIGH_FULL: f64 = 1250.0;
    const HIGH_ZERO: f64 = 1950.0;
    if r <= LOW_ZERO || r >= HIGH_ZERO {
        0.0
    } else if r < LOW_FULL {
        (r - LOW_ZERO) / (LOW_FULL - LOW_ZERO)
    } else if r > HIGH_FULL {
        (HIGH_ZERO - r) / (HIGH_ZERO - HIGH_FULL)
    } else {
        1.0
    }
}

/// Share of played time in the register of a tonic at `tonic_cents`
/// (absolute). For octave-repeating maqamat `tonic_cents` is a pitch class
/// and the best octave is used: the one a listener would hear as the qarar.
fn register_share(notes: &[Prepared], tonic_cents: f64, repeating: bool) -> f64 {
    let total: f64 = notes.iter().map(|n| n.weight).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let at = |tonic: f64| -> f64 {
        notes
            .iter()
            .map(|n| n.weight * register_credit(n.cents - tonic))
            .sum::<f64>()
            / total
    };
    if !repeating {
        return at(tonic_cents);
    }
    let low = notes.iter().map(|n| n.cents).fold(f64::INFINITY, f64::min);
    let high = notes
        .iter()
        .map(|n| n.cents)
        .fold(f64::NEG_INFINITY, f64::max);
    let first = ((low - tonic_cents - 1950.0) / 1200.0).floor() as i64;
    let last = ((high - tonic_cents + 750.0) / 1200.0).ceil() as i64;
    (first..=last)
        .map(|k| at(tonic_cents + 1200.0 * k as f64))
        .fold(0.0, f64::max)
}

/// Number of distinct quarter-tone pitch classes.
fn distinct_classes(notes: &[Prepared]) -> usize {
    let mut classes: Vec<i64> = notes
        .iter()
        .map(|n| ((n.cents / 50.0).round() as i64).rem_euclid(24))
        .collect();
    classes.sort_unstable();
    classes.dedup();
    classes.len()
}

/// Candidate tonic pitch classes (quarter-tone resolution): the
/// most-played classes.
fn candidate_tonics(notes: &[Prepared]) -> Vec<NoteSpelling> {
    let mut weight = [0.0f64; 24];
    for n in notes {
        weight[((n.cents / 50.0).round() as i64).rem_euclid(24) as usize] += n.weight;
    }
    let mut classes: Vec<usize> = (0..24).filter(|&k| weight[k] > 0.0).collect();
    classes.sort_by(|&a, &b| weight[b].total_cmp(&weight[a]));
    classes.truncate(6);
    // Spell each class with its natural/quarter-tone/flat preference from
    // the default table; only the pitch class matters for scoring.
    let table = crate::tuning::SpellingTable::default_24();
    classes.into_iter().map(|k| table.get(k)).collect()
}

/// Score every maqam, tonic, alternative and (for nonperiodic maqamat)
/// register; best first.
fn rank_all(
    notes: &[Prepared],
    now_s: f64,
    dataset: &MaqamDataset,
    tonics: &[NoteSpelling],
    config: &AnalyzerConfig,
) -> Vec<MaqamCandidate> {
    let cadences = cadences(notes, now_s);
    let input = Input {
        notes,
        cadences: &cadences,
    };
    let mut out = Vec::new();
    for maqam in &dataset.maqamat {
        for &t in tonics {
            for variant in
                std::iter::once(None).chain(maqam.variants.iter().map(|v| Some(v.id.as_str())))
            {
                if maqam.octave_repeating {
                    out.extend(score(input, dataset, maqam, t, variant, None, config));
                } else {
                    let low = notes.iter().map(|n| n.cents).fold(f64::INFINITY, f64::min);
                    let high = notes
                        .iter()
                        .map(|n| n.cents)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let first = ((low / 1200.0).floor() as i32 - 3).max(-128);
                    let last = ((high / 1200.0).floor() as i32).min(125);
                    for octave in first..=last {
                        out.extend(score(
                            input,
                            dataset,
                            maqam,
                            t,
                            variant,
                            Some(octave as i8),
                            config,
                        ));
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

/// Keep the best tonic per maqam (input sorted best first) so one maqam
/// does not crowd the list.
fn dedup_by_maqam(candidates: &mut Vec<MaqamCandidate>) {
    let mut seen = Vec::new();
    candidates.retain(|c| {
        if seen.contains(&c.maqam) {
            false
        } else {
            seen.push(c.maqam.clone());
            true
        }
    });
}

/// The material one analysis scores.
#[derive(Clone, Copy)]
struct Input<'a> {
    notes: &'a [Prepared],
    cadences: &'a [Cadence],
}

/// How played pitches are compared with one maqam on one tonic.
#[derive(Debug, Clone, Copy)]
struct Frame {
    /// Tonic: pitch class (periodic) or absolute pitch (nonperiodic), cents.
    tonic_cents: f64,
    repeating: bool,
    /// Documented span of a nonperiodic illustration, cents above the tonic.
    span: (f64, f64),
    /// Tolerance around the span before notes are folded into it.
    margin: f64,
}

impl Frame {
    fn new(
        maqam: &Maqam,
        tonic: NoteSpelling,
        tonic_octave: Option<i8>,
        span: (f64, f64),
        config: &AnalyzerConfig,
    ) -> Self {
        let tonic_cents = tonic_octave.map_or_else(
            || tonic.quarter_tone_class() as f64 * 50.0,
            |octave| {
                SpelledNote::new(tonic.degree, tonic.accidental, octave)
                    .nominal_pitch()
                    .0
                    * 100.0
            },
        );
        Self {
            tonic_cents,
            repeating: maqam.octave_repeating,
            span,
            margin: 1.5 * config.tolerance_cents,
        }
    }

    /// Played pitch relative to the tonic: within the octave for periodic
    /// maqamat; folded into the documented span by octaves otherwise.
    fn rel(&self, cents: f64) -> f64 {
        let r = cents - self.tonic_cents;
        let (lo, hi) = (self.span.0 - self.margin, self.span.1 + self.margin);
        if self.repeating {
            r.rem_euclid(1200.0)
        } else if r < lo {
            r + 1200.0 * ((lo - r) / 1200.0).ceil()
        } else if r > hi {
            r - 1200.0 * ((r - hi) / 1200.0).ceil()
        } else {
            r
        }
    }

    /// A target `cents` above the tonic in the same terms as [`Self::rel`].
    fn target(&self, cents: f64) -> f64 {
        if self.repeating {
            cents.rem_euclid(1200.0)
        } else {
            cents
        }
    }

    fn distance(&self, a: f64, b: f64) -> f64 {
        if self.repeating {
            circ(a, b)
        } else {
            (a - b).abs()
        }
    }
}

/// Circular distance in cents within an octave.
fn circ(a: f64, b: f64) -> f64 {
    let d = a - b;
    // Both are usually already within one octave; then this equals
    // rem_euclid exactly, without the fmod call.
    let d = if d > -1200.0 && d < 0.0 {
        d + 1200.0
    } else if (0.0..1200.0).contains(&d) {
        d
    } else {
        d.rem_euclid(1200.0)
    };
    d.min(1200.0 - d)
}

/// Soft membership of a pitch `distance` cents from a target accepted with
/// `tolerance`: 1 up to half the tolerance, ½ at the tolerance, raised-cosine
/// fade to 0 at 1.5 × the tolerance.
pub fn membership(distance: f64, tolerance: f64) -> f64 {
    let d = distance.abs();
    let lo = 0.5 * tolerance;
    let hi = 1.5 * tolerance;
    if d <= lo {
        1.0
    } else if d >= hi {
        0.0
    } else {
        0.5 * (1.0 + (std::f64::consts::PI * (d - lo) / (hi - lo)).cos())
    }
}

/// A scale degree as a scoring target.
#[derive(Debug, Clone, Copy)]
struct Target {
    /// Cents above the tonic (pitch class for periodic maqamat).
    cents: f64,
    /// Accepted error for this degree.
    tolerance: f64,
}

fn targets(degrees: &[ScaleDegree], repeating: bool, tolerance: f64) -> Vec<Target> {
    let mut v: Vec<f64> = degrees
        .iter()
        .map(|d| {
            if repeating {
                d.cents.rem_euclid(1200.0)
            } else {
                d.cents
            }
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() < 1.0);
    v.into_iter()
        .map(|cents| Target { cents, tolerance })
        .collect()
}

fn score(
    input: Input<'_>,
    dataset: &MaqamDataset,
    maqam: &Maqam,
    tonic: NoteSpelling,
    variant: Option<&str>,
    tonic_octave: Option<i8>,
    config: &AnalyzerConfig,
) -> Option<MaqamCandidate> {
    let notes = input.notes;
    let w = config.weights;
    let repeating = maqam.octave_repeating;
    let scale = maqam.scale(variant).expect("variant comes from this maqam");
    let placements = maqam
        .placements(variant)
        .expect("variant comes from this maqam");
    let mut degrees = scale.to_vec();
    if variant.is_none() {
        if let Some(desc) = &maqam.descending {
            degrees.extend(desc.iter().copied());
        }
    }
    let tol = config.tolerance_cents;
    let classes = targets(&degrees, repeating, tol);
    // Nonperiodic illustrations document a register. Notes inside it are
    // compared exactly, so register-specific degrees stay distinct. Notes
    // outside it (for example below the tonic) are compared, an octave at a
    // time, with the nearest documented register: the data says nothing
    // more specific about them. An anchor is only considered when most of
    // the played time lies inside its documented register, so folding can
    // never move a whole melody into a different register.
    let span = (
        classes.first().map_or(0.0, |c| c.cents),
        classes.last().map_or(0.0, |c| c.cents),
    );
    let frame = Frame::new(maqam, tonic, tonic_octave, span, config);
    let tonic_cents = frame.tonic_cents;
    let margin = frame.margin;
    if !repeating {
        let total: f64 = notes.iter().map(|n| n.weight).sum();
        let inside: f64 = notes
            .iter()
            .filter(|n| {
                let r = n.cents - tonic_cents;
                r >= span.0 - margin && r <= span.1 + margin
            })
            .map(|n| n.weight)
            .sum();
        if inside < 0.5 * total {
            return None;
        }
    }
    // Each note's pitch relative to the tonic, folded once: the loops below
    // compare every note with many targets.
    let rels: Vec<f64> = notes.iter().map(|n| frame.rel(n.cents)).collect();
    let distance = |a: f64, b: f64| frame.distance(a, b);
    // Best matching target and its membership.
    let best_match = |r: f64, targets: &[Target]| -> Option<(usize, f64)> {
        targets
            .iter()
            .enumerate()
            .map(|(i, t)| (i, membership(distance(r, t.cents), t.tolerance)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
    };
    let on = |r: f64, c: f64| membership(distance(r, c), tol);
    let total_time: f64 = notes.iter().map(|n| n.weight).sum();
    // A degree counts as heard after this much (membership-weighted) time,
    // so a single transient does not complete a jins.
    let heard_s = 0.12;

    let mut in_time = 0.0;
    let mut foreign_time = 0.0;
    let mut foreign = Vec::new();
    let mut degree_time = vec![0.0; classes.len()];
    let mut tonic_time = 0.0;
    for (n, &r) in notes.iter().zip(&rels) {
        let m = best_match(r, &classes).map_or(0.0, |(i, m)| {
            degree_time[i] += n.weight * m;
            m
        });
        in_time += n.weight * m;
        foreign_time += n.weight * (1.0 - m);
        if m < 0.5 {
            let q = (n.cents / 50.0).round() / 2.0;
            if !foreign.contains(&q) {
                foreign.push(q);
            }
        }
        tonic_time += n.weight * on(r, 0.0);
    }
    let heard = degree_time.iter().filter(|&&t| t >= heard_s).count();
    let fit = in_time / total_time;
    let foreign_share = foreign_time / total_time;
    let coverage = heard as f64 / classes.len().max(1) as f64;
    let tonic_share = tonic_time / total_time;

    // The latest note that lasted at least as long as the median duration.
    let mut durations: Vec<f64> = notes.iter().map(|n| n.duration).collect();
    durations.sort_by(f64::total_cmp);
    let median = durations[durations.len() / 2];
    let final_rest = notes
        .iter()
        .zip(&rels)
        .rev()
        .find(|(n, _)| n.duration >= median)
        .is_some_and(|(_, &r)| on(r, 0.0) >= 0.5);

    let resting: Vec<f64> = maqam
        .resting_steps
        .iter()
        .filter_map(|s| scale.iter().find(|d| d.step == *s))
        .map(|d| {
            if repeating {
                d.cents.rem_euclid(1200.0)
            } else {
                d.cents
            }
        })
        .collect();
    let resting_time: f64 = notes
        .iter()
        .zip(&rels)
        .map(|(n, &r)| n.weight * resting.iter().map(|&c| on(r, c)).fold(0.0, f64::max))
        .sum();
    let resting_share = if resting.is_empty() {
        0.0
    } else {
        resting_time / total_time
    };

    let mut supporting = vec![
        Evidence::ScaleFit { share: fit },
        Evidence::Coverage {
            heard,
            total: classes.len(),
        },
        Evidence::Tonic {
            share: tonic_share,
            final_rest,
        },
    ];
    if !resting.is_empty() {
        supporting.push(Evidence::RestingTones {
            share: resting_share,
        });
    }

    // Ajnas completeness.
    let mut ajnas_score = 0.0;
    let mut ajnas_count = 0;
    let mut lower_completeness: f64 = 0.0;
    for p in placements {
        let Some(jins) = dataset.jins(&p.jins) else {
            continue;
        };
        let Some(base) = scale.iter().find(|d| d.step == p.on_step) else {
            continue;
        };
        let got = jins
            .degrees
            .iter()
            .map(|d| {
                let c = base.cents + d.cents;
                if repeating { c.rem_euclid(1200.0) } else { c }
            })
            .filter(|&c| {
                notes
                    .iter()
                    .zip(&rels)
                    .map(|(n, &r)| n.weight * on(r, c))
                    .sum::<f64>()
                    >= heard_s
            })
            .count();
        let completeness = got as f64 / jins.degrees.len().max(1) as f64;
        ajnas_score += completeness;
        ajnas_count += 1;
        if p.role == JinsRole::Lower {
            lower_completeness = lower_completeness.max(completeness);
        }
        supporting.push(Evidence::Jins {
            jins: jins.id.clone(),
            completeness,
        });
    }
    let ajnas_avg = if ajnas_count > 0 {
        ajnas_score / ajnas_count as f64
    } else {
        0.0
    };

    // Phrases: subsequence of scale steps (tonic-relative) in the melody.
    let step_of = |n: &Prepared| -> Option<i32> {
        let r = n.cents - tonic_cents;
        if !repeating {
            return scale
                .iter()
                .find(|d| on(r, d.cents) >= 0.5)
                .map(|d| i32::from(d.step));
        }
        let octave = (r / 1200.0).floor() as i32;
        let within = r.rem_euclid(1200.0);
        scale
            .iter()
            .filter(|d| d.step < 7)
            .find(|d| on(within, d.cents) >= 0.5)
            .map(|d| d.step as i32 + 7 * octave)
    };
    let steps: Vec<Option<i32>> = notes.iter().map(step_of).collect();
    let mut phrases_found = 0;
    for (i, ph) in maqam.phrases.iter().enumerate() {
        if ph.steps.len() < 2 {
            continue;
        }
        let pattern: Vec<i32> = ph.steps.iter().map(|&s| s as i32).collect();
        let found = steps.windows(pattern.len()).any(|win| {
            let base = match win[0] {
                Some(b) => b - pattern[0],
                None => return false,
            };
            win.iter().zip(&pattern).all(|(s, p)| {
                *s == Some(p + base) && (if repeating { base % 7 == 0 } else { base == 0 })
            })
        });
        if found {
            phrases_found += 1;
            supporting.push(Evidence::Phrase { index: i });
        }
    }

    // Phrase endings on the tonic.
    let cadence_total: f64 = input.cadences.iter().map(|c| c.strength).sum();
    let cadence_share = if cadence_total > 0.0 {
        input
            .cadences
            .iter()
            .map(|c| c.strength * on(frame.rel(c.cents), 0.0))
            .sum::<f64>()
            / cadence_total
    } else {
        0.0
    };
    if !input.cadences.is_empty() {
        supporting.push(Evidence::Cadence {
            share: cadence_share,
            count: input.cadences.len(),
        });
    }
    let register = register_share(notes, tonic_cents, repeating);
    supporting.push(Evidence::Register { share: register });

    let mut contradicting = Vec::new();
    if foreign_share > 1e-9 {
        contradicting.push(Evidence::ForeignNotes {
            share: foreign_share,
            pitches: foreign,
        });
    }

    let score = w.fit * fit - w.foreign * foreign_share
        + w.coverage * coverage
        + w.tonic * (tonic_share * 2.0).min(1.0)
        + if final_rest { w.final_rest } else { 0.0 }
        + w.resting * resting_share
        + w.ajnas * ajnas_avg
        + w.lower_jins * lower_completeness
        + w.phrase * phrases_found.min(3) as f64
        + w.cadence * cadence_share
        + w.register * register;

    Some(MaqamCandidate {
        maqam: maqam.id.clone(),
        tonic,
        variant: variant.map(str::to_owned),
        tonic_octave,
        score,
        supporting,
        contradicting,
    })
}

/// How well a passage matches one jins.
#[derive(Debug, Clone, Copy)]
struct JinsMatch {
    /// Share of played time on the jins's notes.
    fit: f64,
    /// Ranking score: fit, plus coverage of the jins and emphasis on its
    /// tonal centre.
    score: f64,
    /// The jins's root or centre was heard.
    anchored: bool,
    /// Share of the `outside` time (see [`jins_match`]) on the jins's notes.
    outside_fit: f64,
}

/// Match `recent` with a jins whose notes are `targets` and tonal centre
/// `centre`, all in the terms of `frame`. `outside` optionally gives, per
/// note, the time it lies outside some other scale.
fn jins_match(
    recent: &[Prepared],
    frame: &Frame,
    targets: &[f64],
    centre: f64,
    tolerance: f64,
    outside: &[f64],
) -> JinsMatch {
    let total: f64 = recent.iter().map(|n| n.weight).sum::<f64>().max(1e-9);
    let mut heard = vec![0.0; targets.len()];
    let (mut fit, mut centre_time, mut outside_fit) = (0.0, 0.0, 0.0);
    for (i, n) in recent.iter().enumerate() {
        let r = frame.rel(n.cents);
        let mut best: f64 = 0.0;
        for (h, &t) in heard.iter_mut().zip(targets) {
            let m = membership(frame.distance(r, t), tolerance);
            *h += n.weight * m;
            best = best.max(m);
        }
        fit += n.weight * best;
        centre_time += n.weight * membership(frame.distance(r, centre), tolerance);
        outside_fit += outside.get(i).copied().unwrap_or(0.0) * best;
    }
    let outside_total: f64 = outside.iter().sum();
    let heard_s = 0.12;
    let coverage =
        heard.iter().filter(|&&h| h >= heard_s).count() as f64 / targets.len().max(1) as f64;
    let fit = fit / total;
    JinsMatch {
        fit,
        score: fit + 0.4 * coverage + 0.2 * centre_time / total,
        anchored: heard.first().is_some_and(|&h| h >= heard_s) || centre_time >= heard_s,
        outside_fit: if outside_total > 0.0 {
            outside_fit / outside_total
        } else {
            0.0
        },
    }
}

/// Spell the note `step` letters above `root` along `degrees`.
fn spell_step(root: SpelledNote, degrees: &[ScaleDegree], step: u8) -> Option<SpelledNote> {
    super::realize(root, degrees)
        .into_iter()
        .find(|r| r.step == step)
        .map(|r| r.note)
}

fn spelling(note: SpelledNote) -> NoteSpelling {
    NoteSpelling::new(note.degree, note.accidental)
}

/// Read the most recent passage as a jins: the placement of the leading
/// maqam that explains it best, and, when the maqam's scale does not
/// explain it, a jins outside the maqam that does.
fn read_jins(
    recent: &[Prepared],
    dataset: &MaqamDataset,
    leader: &MaqamCandidate,
    config: &AnalyzerConfig,
) -> (Option<ActiveJins>, Option<DetectedJins>) {
    /// The maqam explains less than this share of the passage (one
    /// characteristic note, such as Saba's diminished fourth, can be all
    /// that leaves a maqam) ...
    const OUTSIDE: f64 = 0.9;
    /// ... and another jins explains at least this share of the notes
    /// outside the maqam, and the passage as well as the maqam does: a
    /// possible modulation.
    const EXPLAINED: f64 = 0.7;
    if recent.len() < 2 || distinct_classes(recent) < 2 {
        return (None, None);
    }
    let Some(maqam) = dataset.maqam(&leader.maqam) else {
        return (None, None);
    };
    let variant = leader.variant.as_deref();
    let (Ok(scale), Ok(placements)) = (maqam.scale(variant), maqam.placements(variant)) else {
        return (None, None);
    };
    let tol = config.tolerance_cents;
    let mut degrees = scale.to_vec();
    if variant.is_none() {
        if let Some(desc) = &maqam.descending {
            degrees.extend(desc.iter().copied());
        }
    }
    let classes = targets(&degrees, maqam.octave_repeating, tol);
    let span = (
        classes.first().map_or(0.0, |c| c.cents),
        classes.last().map_or(0.0, |c| c.cents),
    );
    let frame = Frame::new(maqam, leader.tonic, leader.tonic_octave, span, config);
    let tonic_note = SpelledNote::new(
        leader.tonic.degree,
        leader.tonic.accidental,
        leader.tonic_octave.unwrap_or(4),
    );

    let mut active: Option<(f64, ActiveJins)> = None;
    for p in placements {
        let Some(jins) = dataset.jins(&p.jins) else {
            continue;
        };
        let Some(base) = scale.iter().find(|d| d.step == p.on_step) else {
            continue;
        };
        let jins_targets: Vec<f64> = jins
            .degrees
            .iter()
            .map(|d| frame.target(base.cents + d.cents))
            .collect();
        let centre = centre_cents(jins);
        let m = jins_match(
            recent,
            &frame,
            &jins_targets,
            frame.target(base.cents + centre),
            tol,
            &[],
        );
        if active.as_ref().is_some_and(|(s, _)| *s >= m.score) {
            continue;
        }
        let Some(root) = spell_step(tonic_note, scale, p.on_step) else {
            continue;
        };
        let centre_note = spell_step(root, &jins.degrees, jins.tonic_step).unwrap_or(root);
        active = Some((
            m.score,
            ActiveJins {
                jins: jins.id.clone(),
                on_step: p.on_step,
                role: p.role.clone(),
                root: spelling(root),
                centre: spelling(centre_note),
                fit: m.fit,
            },
        ));
    }

    // How much of the passage the maqam's scale explains at all, and the
    // time of each note outside it.
    let total: f64 = recent.iter().map(|n| n.weight).sum::<f64>().max(1e-9);
    let inside: Vec<f64> = recent
        .iter()
        .map(|n| {
            let r = frame.rel(n.cents);
            classes
                .iter()
                .map(|t| membership(frame.distance(r, t.cents), tol))
                .fold(0.0, f64::max)
        })
        .collect();
    let maqam_fit = recent
        .iter()
        .zip(&inside)
        .map(|(n, m)| n.weight * m)
        .sum::<f64>()
        / total;
    let outside: Vec<f64> = recent
        .iter()
        .zip(&inside)
        .map(|(n, m)| n.weight * (1.0 - m))
        .collect();
    let mut foreign = None;
    if maqam_fit < OUTSIDE {
        let mut roots: Vec<i64> = recent
            .iter()
            .map(|n| ((n.cents / 50.0).round() as i64).rem_euclid(24))
            .collect();
        roots.sort_unstable();
        roots.dedup();
        let table = crate::tuning::SpellingTable::default_24();
        // Ajnas of every documented route of the maqam on this tonic: moving
        // into one of them (Rast descending through Nahawand on Sol) is the
        // maqam's own alternative, not a modulation.
        let tonic_class = leader.tonic.quarter_tone_class() as f64 * 50.0;
        let own: Vec<(&str, i64)> =
            std::iter::once((maqam.ascending.as_slice(), maqam.ajnas.as_slice()))
                .chain(
                    maqam
                        .variants
                        .iter()
                        .map(|v| (v.ascending.as_slice(), v.ajnas.as_slice())),
                )
                .flat_map(|(scale, placements)| {
                    placements.iter().filter_map(move |p| {
                        let base = scale.iter().find(|d| d.step == p.on_step)?;
                        let class =
                            (((tonic_class + base.cents) / 50.0).round() as i64).rem_euclid(24);
                        Some((p.jins.as_str(), class))
                    })
                })
                .collect();
        let mut best: Option<(JinsMatch, &Jins, i64)> = None;
        for jins in &dataset.ajnas {
            let jins_targets: Vec<f64> = jins
                .degrees
                .iter()
                .map(|d| d.cents.rem_euclid(1200.0))
                .collect();
            let centre = centre_cents(jins).rem_euclid(1200.0);
            for &root in &roots {
                let frame = Frame {
                    tonic_cents: root as f64 * 50.0,
                    repeating: true,
                    span: (0.0, 0.0),
                    margin: 0.0,
                };
                let m = jins_match(recent, &frame, &jins_targets, centre, tol, &outside);
                let explains = m.outside_fit >= EXPLAINED && m.fit >= maqam_fit;
                if m.anchored && explains && best.as_ref().is_none_or(|(b, _, _)| m.score > b.score)
                {
                    best = Some((m, jins, root));
                }
            }
        }
        let own_route = best
            .as_ref()
            .is_some_and(|(_, jins, root)| own.contains(&(jins.id.as_str(), *root)));
        if let Some((m, jins, root)) = best.filter(|_| !own_route) {
            let root = table.get(root as usize);
            let root_note = SpelledNote::new(root.degree, root.accidental, 4);
            let centre = spell_step(root_note, &jins.degrees, jins.tonic_step).unwrap_or(root_note);
            foreign = Some(DetectedJins {
                jins: jins.id.clone(),
                root,
                centre: spelling(centre),
                fit: m.fit,
            });
        }
    }
    (active.map(|(_, a)| a), foreign)
}

/// Cents of a jins's tonal centre above its lowest note.
fn centre_cents(jins: &Jins) -> f64 {
    jins.degrees
        .iter()
        .find(|d| d.step == jins.tonic_step)
        .map_or(0.0, |d| d.cents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_distance_matches_rem_euclid() {
        let reference = |a: f64, b: f64| {
            let d = (a - b).rem_euclid(1200.0);
            d.min(1200.0 - d)
        };
        let values = [
            -2400.5, -1200.0, -1199.9, -600.0, -0.0, 0.0, 1e-12, 50.0, 599.5, 1199.99, 1200.0,
            1850.25, 3600.0,
        ];
        for &a in &values {
            for &b in &values {
                assert_eq!(circ(a, b).to_bits(), reference(a, b).to_bits(), "{a} {b}");
            }
        }
    }
    use crate::notation::{Accidental, Degree};

    const FIXTURE: &str = include_str!("../../tests/fixtures/synthetic-maqam-dataset.json");

    fn dataset() -> MaqamDataset {
        MaqamDataset::from_json(FIXTURE).unwrap()
    }

    fn analyzer() -> MaqamAnalyzer {
        MaqamAnalyzer::new(AnalyzerConfig::default())
    }

    /// Push a melody of (pitch, seconds), back to back, starting at t0.
    fn play(a: &mut MaqamAnalyzer, t0: f64, melody: &[(f64, f64)]) -> f64 {
        let mut t = t0;
        for &(pitch, dur) in melody {
            a.push(HistoryNote {
                start_s: t,
                duration_s: dur,
                pitch,
            });
            t += dur;
        }
        t
    }

    fn sp(d: Degree, q: i8) -> NoteSpelling {
        NoteSpelling::new(d, Accidental::new(q).unwrap())
    }

    // SYNTHETIC "TEST maqam A" on Do: 0 200 350 500 700 900 1050 1200 cents
    // → pitches (from Do4 = 60): 60 62 63.5 65 67 69 70.5 72.
    const A_ON_DO: [f64; 8] = [60.0, 62.0, 63.5, 65.0, 67.0, 69.0, 70.5, 72.0];

    #[test]
    fn partial_weights_keep_the_other_defaults() {
        // A field-level `serde(default)` would read a missing weight as 0.
        let w: ScoreWeights = serde_json::from_str(r#"{"tonic":0.5}"#).unwrap();
        assert_eq!(
            w,
            ScoreWeights {
                tonic: 0.5,
                ..ScoreWeights::default()
            }
        );
        let c: AnalyzerConfig = serde_json::from_str(r#"{"weights":{"fit":2.0}}"#).unwrap();
        assert_eq!(c.weights.memory, ScoreWeights::default().memory);
        assert_eq!(c.weights.cadence, ScoreWeights::default().cadence);
        assert_eq!(c.performance_window_s, default_performance_window());
    }

    #[test]
    fn insufficient_data_is_reported() {
        let mut a = analyzer();
        let end = play(&mut a, 0.0, &[(60.0, 0.5), (62.0, 0.5), (63.5, 0.5)]);
        let r = a.analyze(Some(&dataset()), None, end);
        assert_eq!(r.status, AnalysisStatus::InsufficientData);
        assert!(r.candidates.is_empty());
    }

    #[test]
    fn no_dataset_gives_unknown_not_a_guess() {
        let mut a = analyzer();
        let melody: Vec<(f64, f64)> = A_ON_DO.iter().map(|&p| (p, 0.4)).collect();
        let end = play(&mut a, 0.0, &melody);
        let r = a.analyze(None, None, end);
        assert_eq!(r.status, AnalysisStatus::Unknown);
        assert!(r.candidates.is_empty());
    }

    #[test]
    fn scale_material_ranks_its_maqam_first() {
        let mut a = analyzer();
        let mut melody: Vec<(f64, f64)> = A_ON_DO.iter().map(|&p| (p, 0.4)).collect();
        melody.extend([
            (67.0, 0.6),
            (65.0, 0.4),
            (63.5, 0.4),
            (62.0, 0.4),
            (60.0, 1.2),
        ]);
        let end = play(&mut a, 0.0, &melody);
        let r = a.analyze(Some(&dataset()), None, end);
        assert_eq!(r.status, AnalysisStatus::Likely, "{r:#?}");
        let best = &r.candidates[0];
        assert_eq!(best.maqam, "test-maqam-a");
        assert_eq!(best.tonic, sp(Degree::Do, 0));
        assert!(best.contradicting.is_empty());
    }

    #[test]
    fn foreign_notes_are_contradicting_evidence() {
        let mut a = analyzer();
        // Maqam A on Do but with Mi natural (64) instead of Mi half-flat.
        let melody: Vec<(f64, f64)> = [60.0, 62.0, 64.0, 65.0, 67.0, 64.0, 62.0, 60.0]
            .iter()
            .map(|&p| (p, 0.5))
            .collect();
        let end = play(&mut a, 0.0, &melody);
        let r = a.analyze(Some(&dataset()), Some(sp(Degree::Do, 0)), end);
        let c = r
            .candidates
            .iter()
            .find(|c| c.maqam == "test-maqam-a")
            .unwrap();
        assert!(c.contradicting.iter().any(
            |e| matches!(e, Evidence::ForeignNotes { pitches, .. } if pitches.contains(&64.0))
        ));
    }

    #[test]
    fn shared_notes_without_distinguishing_degrees_are_ambiguous() {
        // A dataset with two maqamat that differ only in a degree that is
        // never played: the result must be ambiguous, not a forced pick.
        let mut ds = dataset();
        let mut twin = ds.maqamat[0].clone();
        twin.id = "test-maqam-a-twin".into();
        twin.related.clear();
        twin.modulations.clear();
        twin.ascending[6].cents = 1000.0; // differs only at step 6
        ds.maqamat.push(twin);
        ds.validate().unwrap();

        let mut a = analyzer();
        let melody: Vec<(f64, f64)> = [60.0, 62.0, 63.5, 65.0, 67.0, 65.0, 63.5, 62.0, 60.0]
            .iter()
            .map(|&p| (p, 0.5))
            .collect();
        let end = play(&mut a, 0.0, &melody);
        let r = a.analyze(Some(&ds), Some(sp(Degree::Do, 0)), end);
        assert_eq!(r.status, AnalysisStatus::Ambiguous, "{r:#?}");
    }

    #[test]
    fn fixed_tonic_changes_interpretation() {
        let mut a = analyzer();
        let melody: Vec<(f64, f64)> = A_ON_DO.iter().map(|&p| (p, 0.4)).collect();
        let end = play(&mut a, 0.0, &melody);
        let on_do = a.analyze(Some(&dataset()), Some(sp(Degree::Do, 0)), end);
        let on_re = a.analyze(Some(&dataset()), Some(sp(Degree::Re, 0)), end);
        let s = |r: &MaqamAnalysis| {
            r.candidates
                .iter()
                .find(|c| c.maqam == "test-maqam-a")
                .unwrap()
                .score
        };
        assert!(s(&on_do) > s(&on_re) + 0.3);
        assert_eq!(on_re.tonic_selection, Some(sp(Degree::Re, 0)));
    }

    #[test]
    fn rolling_window_forgets_old_notes() {
        let mut a = MaqamAnalyzer::new(AnalyzerConfig {
            window_s: 5.0,
            ..AnalyzerConfig::default()
        });
        let melody: Vec<(f64, f64)> = A_ON_DO.iter().map(|&p| (p, 0.5)).collect();
        let end = play(&mut a, 0.0, &melody);
        assert_eq!(a.analyze(Some(&dataset()), None, end).notes, 8);
        let r = a.analyze(Some(&dataset()), None, end + 10.0);
        assert_eq!(r.notes, 0);
        assert_eq!(r.status, AnalysisStatus::InsufficientData);
    }

    #[test]
    fn transposition_uses_player_pitch_by_default() {
        use crate::interpret::PitchContext;
        use crate::pitch::{IntervalUnit, Pitch, Transposition};
        use crate::tracking::{NoteTracker, TrackerConfig};
        // Player plays maqam A on Do on an instrument tuned −2: the sounding
        // pitches are on Si♭, but the analysis (player space) finds Do.
        let ctx = PitchContext {
            transposition: Transposition::new(-2.0, IntervalUnit::Semitone),
            ..PitchContext::default()
        };
        let mut tracker = NoteTracker::new(TrackerConfig::default(), ctx.clone());
        let mut a = analyzer();
        let mut t = 0.0;
        let hop = 0.01;
        let mut melody = A_ON_DO.to_vec();
        melody.extend([67.0, 65.0, 63.5, 62.0, 60.0]);
        for (k, &player) in melody.iter().enumerate() {
            let sounding = ctx.transposition.player_to_sounding(Pitch(player));
            for i in 0..40 {
                let obs = crate::events::PitchObservation {
                    time_s: t,
                    window_s: 0.085,
                    status: crate::events::ObservationStatus::Voiced,
                    frequency_hz: Some(sounding.to_frequency(ctx.reference)),
                    confidence: 0.95,
                    level_db: -20.0,
                };
                let onset = (i == 0 && k > 0).then_some(t - 0.1);
                let u = tracker.update(&obs, onset);
                if let Some(e) = &u.ended {
                    a.push_event(e, t);
                }
                t += hop;
            }
        }
        if let Some(e) = tracker.flush(t) {
            a.push_event(&e, t);
        }
        let r = a.analyze(Some(&dataset()), None, t);
        assert_eq!(r.candidates[0].maqam, "test-maqam-a");
        assert_eq!(r.candidates[0].tonic, sp(Degree::Do, 0), "{r:#?}");
    }

    /// Play `melody` back to back from `t0`, analysing after every note
    /// the way a live application does; the last analysis.
    fn play_live(
        a: &mut MaqamAnalyzer,
        ds: &MaqamDataset,
        t0: f64,
        melody: &[(f64, f64)],
    ) -> (f64, MaqamAnalysis) {
        let mut t = t0;
        let mut last = None;
        for &(pitch, dur) in melody {
            a.push(HistoryNote {
                start_s: t,
                duration_s: dur,
                pitch,
            });
            t += dur;
            last = Some(a.analyze(Some(ds), None, t));
        }
        (t, last.expect("a melody"))
    }

    /// Maqam A on Do: up its lower jins, resting on the tonic.
    const A_LOWER: [(f64, f64); 9] = [
        (60.0, 0.4),
        (62.0, 0.4),
        (63.5, 0.4),
        (65.0, 0.4),
        (67.0, 0.6),
        (65.0, 0.4),
        (63.5, 0.4),
        (62.0, 0.4),
        (60.0, 1.2),
    ];

    #[test]
    fn a_passage_in_the_upper_jins_is_read_as_that_jins_not_a_modulation() {
        let ds = dataset();
        let mut a = analyzer();
        let mut t = 0.0;
        for _ in 0..2 {
            t = play_live(&mut a, &ds, t, &A_LOWER).0;
        }
        // Dwell on the upper jins (X on Sol: 67 69 70.5 72), resting on Sol.
        let upper = [
            (67.0, 0.4),
            (69.0, 0.4),
            (70.5, 0.4),
            (72.0, 0.6),
            (70.5, 0.4),
            (69.0, 0.4),
            (67.0, 1.0),
        ];
        let mut r = None;
        for _ in 0..3 {
            let (end, last) = play_live(&mut a, &ds, t, &upper);
            t = end;
            r = Some(last);
        }
        let r = r.unwrap();
        assert_eq!(r.candidates[0].maqam, "test-maqam-a", "{r:#?}");
        assert_eq!(r.candidates[0].tonic, sp(Degree::Do, 0));
        let active = r.active_jins.as_ref().expect("an active jins");
        assert_eq!(active.on_step, 4);
        assert_eq!(active.role, JinsRole::Upper);
        assert_eq!(active.root, sp(Degree::Sol, 0));
        assert!(!r.possible_modulation && r.foreign_jins.is_none(), "{r:#?}");
    }

    #[test]
    fn a_passage_outside_the_maqam_is_a_possible_modulation() {
        let ds = dataset();
        let mut a = analyzer();
        let mut t = 0.0;
        for _ in 0..2 {
            t = play_live(&mut a, &ds, t, &A_LOWER).0;
        }
        let (_, before) = play_live(&mut a, &ds, t, &A_LOWER);
        let active = before.active_jins.as_ref().expect("an active jins");
        assert_eq!((active.on_step, &active.role), (0, &JinsRole::Lower));
        t += A_LOWER.iter().map(|n| n.1).sum::<f64>();
        let y = Y_ON_RE;
        // A modulation lasts more than one gesture.
        let (t, _) = play_live(&mut a, &ds, t, &y);
        let (_, r) = play_live(&mut a, &ds, t, &y);
        let foreign = r.foreign_jins.as_ref().unwrap_or_else(|| panic!("{r:#?}"));
        assert_eq!(foreign.jins, "test-jins-y", "{r:#?}");
        assert_eq!(foreign.root, sp(Degree::Re, 0));
        assert!(r.possible_modulation);
    }

    /// Jins Y on Re (0 100 400 500 → 62 63 66 67): not in maqam A.
    const Y_ON_RE: [(f64, f64); 7] = [
        (62.0, 0.4),
        (63.0, 0.4),
        (66.0, 0.4),
        (67.0, 0.6),
        (66.0, 0.4),
        (63.0, 0.4),
        (62.0, 1.0),
    ];

    #[test]
    fn the_performance_keeps_its_maqam_through_a_passage_elsewhere() {
        let ds = dataset();
        let mut a = analyzer();
        let mut t = 0.0;
        for _ in 0..8 {
            t = play_live(&mut a, &ds, t, &A_LOWER).0;
        }
        let mut r = None;
        for _ in 0..4 {
            let (end, last) = play_live(&mut a, &ds, t, &Y_ON_RE);
            t = end;
            r = Some(last);
        }
        let r = r.unwrap();
        // The latest passage has left maqam A ...
        assert_eq!(r.candidates[0].maqam, "test-maqam-b", "{r:#?}");
        // ... the performance as a whole has not.
        let p = r.performance.as_ref().expect("a performance reading");
        assert_eq!(p.candidates[0].maqam, "test-maqam-a", "{p:#?}");
        assert_eq!(p.candidates[0].tonic, sp(Degree::Do, 0));
        assert_eq!(p.status, AnalysisStatus::Likely);
        assert!(p.notes > r.notes);
        assert!(p.window.0 < r.window.0);
    }

    #[test]
    fn the_performance_is_still_read_while_the_latest_passage_is_too_sparse() {
        let ds = dataset();
        let mut a = analyzer();
        let mut t = 0.0;
        for _ in 0..3 {
            t = play_live(&mut a, &ds, t, &A_LOWER).0;
        }
        // Twelve seconds of silence: nothing in the window, same piece.
        let r = a.analyze(Some(&ds), None, t + 12.0);
        assert_eq!(r.status, AnalysisStatus::InsufficientData);
        assert!(r.candidates.is_empty());
        let p = r.performance.expect("a performance reading");
        assert_eq!(p.candidates[0].maqam, "test-maqam-a");
        assert_eq!(p.window.1, t + 12.0);
    }

    #[test]
    fn a_long_silence_starts_a_new_performance() {
        let ds = dataset();
        let mut a = analyzer();
        let mut t = 0.0;
        for _ in 0..3 {
            t = play_live(&mut a, &ds, t, &A_LOWER).0;
        }
        // Still the same piece until something new is played.
        let waiting = a.analyze(Some(&ds), None, t + 25.0);
        assert!(waiting.performance.is_some());
        // The first notes after the silence do not make a reading yet ...
        let (t, r) = play_live(&mut a, &ds, t + 25.0, &Y_ON_RE[..3]);
        assert!(r.performance.is_none(), "{r:#?}");
        // ... and the new piece is read on its own.
        let (t, _) = play_live(&mut a, &ds, t, &Y_ON_RE);
        let (_, r) = play_live(&mut a, &ds, t, &Y_ON_RE);
        let p = r.performance.expect("a performance reading");
        assert_eq!(p.candidates[0].maqam, "test-maqam-b", "{p:#?}");
        assert!(p.window.0 >= 25.0);
    }

    #[test]
    fn the_performance_reading_can_be_switched_off() {
        let ds = dataset();
        let mut a = MaqamAnalyzer::new(AnalyzerConfig {
            performance_window_s: 0.0,
            ..AnalyzerConfig::default()
        });
        let mut t = 0.0;
        let mut r = None;
        for _ in 0..20 {
            let (end, last) = play_live(&mut a, &ds, t, &A_LOWER);
            t = end;
            r = Some(last);
        }
        assert!(r.unwrap().performance.is_none());
        // Nothing older than the tuning window is kept.
        let oldest = a.history().next().unwrap();
        assert!(oldest.start_s + oldest.duration_s >= t - a.config().tuning_window_s);
    }

    #[test]
    fn the_performance_window_bounds_the_history() {
        let ds = dataset();
        let mut a = MaqamAnalyzer::new(AnalyzerConfig {
            performance_window_s: 90.0,
            ..AnalyzerConfig::default()
        });
        let mut t = 0.0;
        let mut r = None;
        for _ in 0..40 {
            let (end, last) = play_live(&mut a, &ds, t, &A_LOWER);
            t = end;
            r = Some(last);
        }
        let p = r.unwrap().performance.unwrap();
        assert!(
            p.window.0 >= t - 90.0 - 1.2 && p.window.0 > 60.0,
            "{:?}",
            p.window
        );
    }

    #[test]
    fn phrase_endings_are_long_tones_or_tones_before_a_pause() {
        let n = |cents: f64, start: f64, duration: f64| Prepared {
            cents,
            weight: duration,
            start,
            duration,
        };
        // Tremolo on 6200 (three plucks), then a pause; a short 6400; a long
        // 6000 approached from above.
        let notes = [
            n(6500.0, 0.0, 0.3),
            n(6200.0, 0.3, 0.2),
            n(6205.0, 0.5, 0.2),
            n(6198.0, 0.7, 0.2),
            n(6400.0, 1.5, 0.3),
            n(6000.0, 1.8, 1.2),
        ];
        let c = cadences(&notes, 3.0);
        assert_eq!(c.len(), 2, "{c:?}");
        assert!((c[0].cents - 6201.0).abs() < 5.0);
        assert!((c[1].cents - 6000.0).abs() < 1e-9);
        assert!(c[1].strength > c[0].strength);
    }

    #[test]
    fn register_prefers_the_tonic_at_the_bottom_of_the_melody() {
        let notes: Vec<Prepared> = [
            6200.0, 6300.0, 6600.0, 6700.0, 6900.0, 7000.0, 7200.0, 7400.0,
        ]
        .iter()
        .enumerate()
        .map(|(i, &cents)| Prepared {
            cents,
            weight: 0.5,
            start: i as f64 * 0.5,
            duration: 0.5,
        })
        .collect();
        let re = register_share(&notes, 200.0, true);
        let sol = register_share(&notes, 700.0, true);
        assert!((re - 1.0).abs() < 1e-9);
        assert!(sol < re - 0.05, "{sol}");
    }

    /// Maqam A on Do, ending on the tonic, every pitch shifted by `cents`.
    fn detuned_melody(cents: f64) -> Vec<(f64, f64)> {
        let mut melody: Vec<(f64, f64)> = A_ON_DO.iter().map(|&p| (p, 0.4)).collect();
        melody.extend([
            (67.0, 0.6),
            (65.0, 0.4),
            (63.5, 0.4),
            (62.0, 0.4),
            (60.0, 1.2),
        ]);
        melody
            .into_iter()
            .map(|(p, d)| (p + cents / 100.0, d))
            .collect()
    }

    #[test]
    fn membership_is_soft_with_half_credit_at_the_tolerance() {
        assert_eq!(membership(0.0, 30.0), 1.0);
        assert_eq!(membership(15.0, 30.0), 1.0);
        assert!((membership(30.0, 30.0) - 0.5).abs() < 1e-12);
        assert!((membership(-30.0, 30.0) - 0.5).abs() < 1e-12);
        assert_eq!(membership(45.0, 30.0), 0.0);
        assert!(membership(25.0, 30.0) > membership(35.0, 30.0));
    }

    #[test]
    fn no_cliff_at_the_tolerance_boundary() {
        // One note 29 vs 31 cents off: the hard 30-cent cutoff used to flip
        // it from "in scale" to "foreign"; now the fit barely changes.
        let fit = |off: f64| {
            let mut a = MaqamAnalyzer::new(AnalyzerConfig {
                tuning_offset_cents: Some(0.0),
                ..AnalyzerConfig::default()
            });
            let mut melody = detuned_melody(0.0);
            melody[2].0 += off / 100.0;
            let end = play(&mut a, 0.0, &melody);
            let r = a.analyze(Some(&dataset()), Some(sp(Degree::Do, 0)), end);
            let c = r
                .candidates
                .iter()
                .find(|c| c.maqam == "test-maqam-a")
                .unwrap()
                .clone();
            c.supporting
                .iter()
                .find_map(|e| match e {
                    Evidence::ScaleFit { share } => Some(*share),
                    _ => None,
                })
                .unwrap()
        };
        assert!((fit(29.0) - fit(31.0)).abs() < 0.01);
    }

    #[test]
    fn tuning_offset_is_found_and_quarter_tones_do_not_pull_it() {
        // Maqam A contains two half-flats; the instrument is 12 cents sharp.
        let notes: Vec<HistoryNote> = detuned_melody(12.0)
            .into_iter()
            .enumerate()
            .map(|(i, (pitch, d))| HistoryNote {
                start_s: i as f64,
                duration_s: d,
                pitch,
            })
            .collect();
        let est = estimate_tuning_offset(&notes, 20.0).unwrap();
        assert!((est - 12.0).abs() < 1.0, "{est}");
        let flat: Vec<HistoryNote> = notes
            .iter()
            .map(|n| HistoryNote {
                pitch: n.pitch - 0.2,
                ..*n
            })
            .collect();
        let est = estimate_tuning_offset(&flat, 20.0).unwrap();
        assert!((est + 8.0).abs() < 1.0, "{est}");
    }

    #[test]
    fn tuning_offset_is_bounded_and_needs_material() {
        let notes: Vec<HistoryNote> = (0..8)
            .map(|i| HistoryNote {
                start_s: i as f64,
                duration_s: 0.5,
                pitch: 60.0 + i as f64 + 0.4,
            })
            .collect();
        let est = estimate_tuning_offset(&notes, 20.0).unwrap();
        assert!(
            est.abs() <= 20.0,
            "a 40-cent shift must not be followed: {est}"
        );
        assert_eq!(estimate_tuning_offset(&notes[..1], 20.0), None);
    }

    #[test]
    fn a_detuned_instrument_still_names_the_right_tonic() {
        // 27 cents sharp: without compensation every note rounds up to the
        // next quarter-tone and the tonic is misnamed as Do half-sharp.
        let run = |offset: Option<f64>| {
            let mut a = MaqamAnalyzer::new(AnalyzerConfig {
                tuning_offset_cents: offset,
                ..AnalyzerConfig::default()
            });
            let end = play(&mut a, 0.0, &detuned_melody(27.0));
            a.analyze(Some(&dataset()), None, end)
        };
        let fixed = run(Some(0.0));
        assert_eq!(fixed.candidates[0].maqam, "test-maqam-a");
        assert_ne!(fixed.candidates[0].tonic, sp(Degree::Do, 0));
        let auto = run(None);
        assert_eq!(auto.candidates[0].maqam, "test-maqam-a", "{auto:#?}");
        assert_eq!(auto.candidates[0].tonic, sp(Degree::Do, 0));
        assert!((auto.tuning_offset_cents - 20.0).abs() < 1e-9);
        assert!(auto.candidates[0].contradicting.is_empty());
    }

    #[test]
    fn short_transients_are_ignored() {
        let mut melody = detuned_melody(0.0);
        // Pluck transients a quarter-tone off, 50 ms each.
        for k in [3, 6, 9] {
            melody.insert(k, (64.5, 0.05));
        }
        let run = |min_note_s: f64| {
            let mut a = MaqamAnalyzer::new(AnalyzerConfig {
                min_note_s,
                ..AnalyzerConfig::default()
            });
            let end = play(&mut a, 0.0, &melody);
            a.analyze(Some(&dataset()), Some(sp(Degree::Do, 0)), end)
        };
        let kept = run(0.0);
        let filtered = run(AnalyzerConfig::default().min_note_s);
        let a_of = |r: &MaqamAnalysis| {
            r.candidates
                .iter()
                .find(|c| c.maqam == "test-maqam-a")
                .unwrap()
                .clone()
        };
        assert!(!a_of(&kept).contradicting.is_empty());
        assert!(a_of(&filtered).contradicting.is_empty());
        assert_eq!(filtered.notes, kept.notes - 3);
    }

    #[test]
    fn old_notes_leave_the_window_but_still_set_the_tuning() {
        let run = |tuning_window_s: f64| {
            let mut a = MaqamAnalyzer::new(AnalyzerConfig {
                tuning_window_s,
                ..AnalyzerConfig::default()
            });
            let mut t = 0.0;
            for _ in 0..3 {
                t = play(&mut a, t, &detuned_melody(15.0));
            }
            // Twenty seconds later, a few notes: too little to tune from.
            let end = play(
                &mut a,
                t + 20.0,
                &[(60.15, 0.3), (62.15, 0.3), (65.15, 0.3)],
            );
            a.analyze(Some(&dataset()), None, end)
        };
        let r = run(60.0);
        assert_eq!(r.notes, 3, "only the last ten seconds are scored");
        assert!((r.tuning_offset_cents - 15.0).abs() < 1.0, "{r:#?}");
        assert_eq!(run(10.0).tuning_offset_cents, 0.0);
    }

    #[test]
    fn a_near_tie_keeps_the_current_maqam() {
        let mut ds = dataset();
        let mut twin = ds.maqamat[0].clone();
        twin.id = "test-maqam-a-twin".into();
        twin.related.clear();
        twin.modulations.clear();
        twin.ascending[6].cents = 1000.0;
        ds.maqamat.push(twin);
        let run = |continuity: f64| {
            let mut a = MaqamAnalyzer::new(AnalyzerConfig {
                weights: ScoreWeights {
                    continuity,
                    // Only the continuity bonus, without the score history.
                    memory: 0.0,
                    ..ScoreWeights::default()
                },
                ..AnalyzerConfig::default()
            });
            // Clearly maqam A (step 6 at 1050 cents) ...
            let t = play(&mut a, 0.0, &detuned_melody(0.0));
            let first = a.analyze(Some(&ds), Some(sp(Degree::Do, 0)), t);
            assert_eq!(first.candidates[0].maqam, "test-maqam-a");
            // ... then a passage whose step 6 (1020 cents) leans to the twin.
            let melody: Vec<(f64, f64)> = detuned_melody(0.0)
                .into_iter()
                .map(|(p, d)| if p == 70.5 { (70.2, 0.3) } else { (p, d) })
                .collect();
            let end = play(&mut a, t + 15.0, &melody);
            a.analyze(Some(&ds), Some(sp(Degree::Do, 0)), end)
                .candidates[0]
                .maqam
                .clone()
        };
        assert_eq!(run(0.0), "test-maqam-a-twin");
        assert_eq!(run(ScoreWeights::default().continuity), "test-maqam-a");
    }

    #[test]
    fn clear_forgets_the_current_maqam() {
        let mut a = analyzer();
        let end = play(&mut a, 0.0, &detuned_melody(0.0));
        a.analyze(Some(&dataset()), None, end);
        assert!(a.leader.is_some());
        a.clear();
        assert!(a.leader.is_none());
    }
}
