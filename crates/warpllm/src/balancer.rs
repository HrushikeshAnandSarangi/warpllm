//! Smooth weighted round-robin load balancing across providers.
//!
//! One [`Balancer`] per model_str that has a `balance:` field. Bounded by the
//! number of balance candidates — a handful, from config — so no unbounded
//! structures grow at runtime.
//!
//! Algorithm: Nginx-style smooth weighted round-robin. Each candidate holds a
//! `current_weight` (atomic, starts at 0). On each selection:
//!
//! 1. Every candidate's `current_weight` is incremented by its effective weight.
//! 2. The candidate with the highest `current_weight` is selected.
//! 3. The selected candidate's `current_weight` is decremented by the total weight.
//!
//! This produces exact distribution over each cycle (length = total weight) with
//! maximum smoothness — no two identical picks are more than
//! `ceil(total / max_weight)` apart.
//!
//! Thread safety: each candidate's `current_weight` is an independent
//! [`AtomicI32`]. Two threads may select the same candidate (benign — one extra
//! request to that provider), but the distribution remains correct over the cycle.

use std::sync::atomic::{AtomicI32, Ordering};

use crate::registry::{ModelSpec, ProviderSpec};

/// One candidate in a balanced model's rotation.
///
/// Resolved from a [`BalanceCandidate`](crate::BalanceCandidate) at client
/// construction: the target `model_str` is looked up in the registry, and the
/// resulting provider/model pair is stored here for the balancer to hand back
/// on each selection.
#[derive(Debug)]
pub(crate) struct Candidate {
    pub(crate) provider: &'static ProviderSpec,
    pub(crate) model: &'static ModelSpec,
    pub(crate) weight: u32,
}

/// Smooth weighted round-robin balancer for one model_str.
///
/// Built once at [`Client`](crate::Client) construction from the `balance:`
/// entries in the roster. The candidate set is static and bounded by the
/// roster, so no runtime allocation beyond the initial construction.
///
/// # Distribution example
///
/// For candidates with weights `[3, 1]` (total = 4):
///
/// ```text
/// Step 1: A=3, B=1  → pick A → A=-1, B=1   → A
/// Step 2: A=2, B=2  → pick A → A=-2, B=2   → A
/// Step 3: A=1, B=3  → pick B → A=1,  B=-1  → B
/// Step 4: A=4, B=0  → pick A → A=0,  B=0   → A
/// ```
///
/// Cycle: A, A, B, A — exactly 75%/25%, perfectly interleaved.
#[derive(Debug)]
pub(crate) struct Balancer {
    candidates: Vec<Candidate>,
    /// Per-candidate current weight. `AtomicI32` because `current_weight`
    /// goes negative during the cycle (e.g., A=3-4=-1 after first pick).
    current: Vec<AtomicI32>,
    /// Total weight across all candidates. Stored once, used on every
    /// decrement step.
    total: i32,
}

impl Balancer {
    /// Build a balancer from a resolved candidate list.
    ///
    /// Candidates must be non-empty — validated at load time by the registry
    /// and at construction time by the client.
    pub(crate) fn new(candidates: Vec<Candidate>) -> Self {
        let total = candidates.iter().map(|c| c.weight as i32).sum();
        let current = candidates.iter().map(|_| AtomicI32::new(0)).collect();
        Self {
            candidates,
            current,
            total,
        }
    }

    /// Select the next candidate via smooth weighted round-robin.
    ///
    /// Lock-free: each candidate's `current_weight` is an independent atomic.
    /// Two threads may select the same candidate (benign — one extra request
    /// to that provider), but the distribution remains correct over the cycle.
    pub(crate) fn select(&self) -> &Candidate {
        // Step 1: increment every candidate's current_weight.
        for (i, c) in self.candidates.iter().enumerate() {
            self.current[i].fetch_add(c.weight as i32, Ordering::Relaxed);
        }
        // Step 2: pick the candidate with the highest current_weight.
        let mut best = i32::MIN;
        let mut best_idx = 0;
        for i in 0..self.candidates.len() {
            let w = self.current[i].load(Ordering::Relaxed);
            if w > best {
                best = w;
                best_idx = i;
            }
        }
        // Step 3: decrement the winner by total weight.
        self.current[best_idx].fetch_sub(self.total, Ordering::Relaxed);
        &self.candidates[best_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(name: &str, weight: u32) -> Candidate {
        // Leaked because the balancer takes `&'static` specs. Costs nothing
        // in a test process that is about to exit.
        let provider = Box::leak(Box::new(ProviderSpec {
            name: name.to_string(),
            base_url: format!("https://api.{name}.test"),
            env_api_key: None,
        }));
        let model = Box::leak(Box::new(ModelSpec {
            provider: name.to_string(),
            model: name.to_string(),
            supported_apis: vec![],
            capabilities: crate::Capabilities::blank(),
            deprecation_date: None,
            balance: None,
        }));
        Candidate {
            provider,
            model,
            weight,
        }
    }

    #[test]
    fn single_candidate_always_selects() {
        let balancer = Balancer::new(vec![candidate("a", 1)]);
        for _ in 0..100 {
            assert_eq!(balancer.select().provider.name(), "a");
        }
    }

    #[test]
    fn two_equal_weight_candidates_alternate() {
        let balancer = Balancer::new(vec![candidate("a", 1), candidate("b", 1)]);
        let picks: Vec<&str> = (0..6).map(|_| balancer.select().provider.name()).collect();
        assert_eq!(picks, vec!["a", "b", "a", "b", "a", "b"]);
    }

    #[test]
    fn three_to_one_ratio() {
        let balancer = Balancer::new(vec![candidate("a", 3), candidate("b", 1)]);
        let mut counts = [0u32; 2];
        for _ in 0..1000 {
            let c = balancer.select();
            if c.provider.name() == "a" {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
        }
        // Exact distribution: 750/250 over each cycle of 4.
        assert_eq!(
            counts[0], 750,
            "weight-3 candidate should be selected 750 times"
        );
        assert_eq!(
            counts[1], 250,
            "weight-1 candidate should be selected 250 times"
        );
    }

    #[test]
    fn distribution_is_exact_over_cycle() {
        // Weights [2, 1, 1], total = 4. Over 4 picks, expect 2/1/1.
        let balancer = Balancer::new(vec![
            candidate("a", 2),
            candidate("b", 1),
            candidate("c", 1),
        ]);
        let mut counts = [0u32; 3];
        for _ in 0..4 {
            let c = balancer.select();
            let idx = match c.provider.name() {
                "a" => 0,
                "b" => 1,
                "c" => 2,
                _ => unreachable!(),
            };
            counts[idx] += 1;
        }
        assert_eq!(counts, [2, 1, 1]);
    }

    #[test]
    fn smoothness_no_two_identical_picks_are_far_apart() {
        // Weights [5, 1], total = 6. Max gap between A picks is ceil(6/5) = 2.
        let balancer = Balancer::new(vec![candidate("a", 5), candidate("b", 1)]);
        let picks: Vec<&str> = (0..12).map(|_| balancer.select().provider.name()).collect();
        // Find gaps between consecutive A picks.
        let a_positions: Vec<usize> = picks
            .iter()
            .enumerate()
            .filter(|(_, name)| **name == "a")
            .map(|(i, _)| i)
            .collect();
        for pair in a_positions.windows(2) {
            let gap = pair[1] - pair[0];
            assert!(
                gap <= 2,
                "gap between A picks should be at most 2, got {gap}: {picks:?}"
            );
        }
    }
}
