//! Atomic Signals with Boolean Predicate Logic
//!
//! # T1 Primitive Grounding
//!
//! | Concept | T1 | Symbol | Rust |
//! |---------|----|----|------|
//! | Signal existence | Existence | ∃ | `bool` |
//! | Composition | Sum | Σ | `SignalExpr` enum |
//! | Evaluation | Comparison | κ | `evaluate()` |
//! | Emission | Causality | → | `emit()` |
//! | Feedback | Recursion | ρ | Result → next eval |
//!
//! # Design Principle
//!
//! Every hook decision reduces to atomic yes/no predicates composed with AND/OR/NOT.
//! This is the irreducible foundation — no simpler decomposition exists.
//!
//! ```text
//! Atom: "Does panic!() exist in content?" → true/false
//! Composition: "panic OR unwrap" → true if either
//! Decision: true → Block, false → Pass
//! ```

use std::fmt;
use std::sync::Arc;

// ============================================================================
// T1 PRIMITIVES: Atomic Signals
// ============================================================================

/// An atomic signal — the irreducible unit of hook logic.
///
/// # Tier: T1
/// Grounds to: Existence (∃) — does this condition exist?
///
/// # Example
/// ```ignore
/// let panic_signal = AtomicSignal::new("panic", |content| {
///     content.contains("panic!")
/// });
/// assert!(panic_signal.evaluate("fn main() { panic!(\"boom\"); }"));
/// ```
#[derive(Clone)]
pub struct AtomicSignal {
    /// Signal identifier
    pub name: &'static str,
    /// Predicate function: content → bool
    predicate: Arc<dyn Fn(&str) -> bool + Send + Sync>,
}

impl AtomicSignal {
    /// Create a new atomic signal with a predicate.
    pub fn new<F>(name: &'static str, predicate: F) -> Self
    where
        F: Fn(&str) -> bool + Send + Sync + 'static,
    {
        Self {
            name,
            predicate: Arc::new(predicate),
        }
    }

    /// Evaluate the predicate against content.
    ///
    /// Returns `true` if the condition exists, `false` otherwise.
    #[inline]
    pub fn evaluate(&self, content: &str) -> bool {
        (self.predicate)(content)
    }

    /// Create a constant true signal (always fires).
    pub fn always(name: &'static str) -> Self {
        Self::new(name, |_| true)
    }

    /// Create a constant false signal (never fires).
    pub fn never(name: &'static str) -> Self {
        Self::new(name, |_| false)
    }
}

impl fmt::Debug for AtomicSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtomicSignal")
            .field("name", &self.name)
            .finish()
    }
}

// ============================================================================
// T2-P PRIMITIVES: Signal Composition
// ============================================================================

/// Composed signal expression — combines atomic signals with boolean logic.
///
/// # Tier: T2-P
/// Grounds to: Sum (Σ) — coproduct of composition operators.
///
/// # Logic
/// - `And(signals)`: All must evaluate true
/// - `Or(signals)`: At least one must evaluate true
/// - `Not(signal)`: Inverts the signal
/// - `Atom(signal)`: Leaf node — evaluates single atomic signal
#[derive(Clone)]
pub enum SignalExpr {
    /// All child expressions must be true (conjunction)
    And(Vec<SignalExpr>),
    /// At least one child must be true (disjunction)
    Or(Vec<SignalExpr>),
    /// Inverts the child expression (negation)
    Not(Box<SignalExpr>),
    /// Leaf node — atomic signal evaluation
    Atom(AtomicSignal),
}

impl SignalExpr {
    /// Create an AND composition of expressions.
    pub fn and(exprs: Vec<SignalExpr>) -> Self {
        Self::And(exprs)
    }

    /// Create an OR composition of expressions.
    pub fn or(exprs: Vec<SignalExpr>) -> Self {
        Self::Or(exprs)
    }

    /// Create a NOT composition (negation).
    #[allow(clippy::should_implement_trait)]
    pub fn not(expr: SignalExpr) -> Self {
        Self::Not(Box::new(expr))
    }

    /// Create an atomic leaf node.
    pub fn atom(signal: AtomicSignal) -> Self {
        Self::Atom(signal)
    }

    /// Evaluate the expression tree against content.
    ///
    /// # T1 Grounding
    /// Uses comparison (κ) to reduce tree to single boolean.
    pub fn evaluate(&self, content: &str) -> bool {
        match self {
            Self::And(exprs) => exprs.iter().all(|e| e.evaluate(content)),
            Self::Or(exprs) => exprs.iter().any(|e| e.evaluate(content)),
            Self::Not(expr) => !expr.evaluate(content),
            Self::Atom(signal) => signal.evaluate(content),
        }
    }

    /// Count the number of atomic signals in the expression.
    pub fn atom_count(&self) -> usize {
        match self {
            Self::And(exprs) | Self::Or(exprs) => exprs.iter().map(|e| e.atom_count()).sum(),
            Self::Not(expr) => expr.atom_count(),
            Self::Atom(_) => 1,
        }
    }

    /// Get all atomic signal names in the expression.
    pub fn signal_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        self.collect_names(&mut names);
        names
    }

    fn collect_names(&self, names: &mut Vec<&'static str>) {
        match self {
            Self::And(exprs) | Self::Or(exprs) => {
                for expr in exprs {
                    expr.collect_names(names);
                }
            }
            Self::Not(expr) => expr.collect_names(names),
            Self::Atom(signal) => names.push(signal.name),
        }
    }
}

impl fmt::Debug for SignalExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::And(exprs) => write!(f, "And({:?})", exprs),
            Self::Or(exprs) => write!(f, "Or({:?})", exprs),
            Self::Not(expr) => write!(f, "Not({:?})", expr),
            Self::Atom(signal) => write!(f, "Atom({})", signal.name),
        }
    }
}

// ============================================================================
// T2-C COMPOSITES: Signal Evaluation Result
// ============================================================================

/// Result of evaluating a signal expression.
///
/// # Tier: T2-C
/// Grounds to: T1(bool) + T2-P(SignalExpr) metadata.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// The boolean outcome
    pub fired: bool,
    /// Which atomic signals fired (for diagnostics)
    pub fired_signals: Vec<&'static str>,
    /// Total atoms evaluated
    pub atoms_evaluated: usize,
}

impl EvalResult {
    /// Create a new evaluation result.
    pub fn new(fired: bool, fired_signals: Vec<&'static str>, atoms_evaluated: usize) -> Self {
        Self {
            fired,
            fired_signals,
            atoms_evaluated,
        }
    }
}

/// Evaluate an expression with detailed tracking.
///
/// Returns both the boolean result and which signals fired.
pub fn evaluate_tracked(expr: &SignalExpr, content: &str) -> EvalResult {
    let mut fired_signals = Vec::new();
    let mut atoms_evaluated = 0;
    let fired = eval_tracked_inner(expr, content, &mut fired_signals, &mut atoms_evaluated);
    EvalResult::new(fired, fired_signals, atoms_evaluated)
}

fn eval_tracked_inner(
    expr: &SignalExpr,
    content: &str,
    fired: &mut Vec<&'static str>,
    count: &mut usize,
) -> bool {
    match expr {
        SignalExpr::And(exprs) => {
            for e in exprs {
                if !eval_tracked_inner(e, content, fired, count) {
                    return false;
                }
            }
            true
        }
        SignalExpr::Or(exprs) => {
            for e in exprs {
                if eval_tracked_inner(e, content, fired, count) {
                    return true;
                }
            }
            false
        }
        SignalExpr::Not(e) => !eval_tracked_inner(e, content, fired, count),
        SignalExpr::Atom(signal) => {
            *count += 1;
            let result = signal.evaluate(content);
            if result {
                fired.push(signal.name);
            }
            result
        }
    }
}

// ============================================================================
// PREDEFINED ATOMIC SIGNALS (Hook Library)
// ============================================================================

/// Standard atomic signals for Rust code analysis.
pub mod signals {
    use super::AtomicSignal;
    use regex::Regex;
    use std::sync::LazyLock;

    // Compile regex patterns once
    static PANIC_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"panic!\s*\(").expect("valid regex"));
    static UNWRAP_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\.unwrap\(\)").expect("valid regex"));
    static EXPECT_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\.expect\(").expect("valid regex"));
    static UNSAFE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\bunsafe\s*\{").expect("valid regex"));
    static TRANSMUTE_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"mem::transmute").expect("valid regex"));
    static TODO_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)todo!|fixme|xxx").expect("valid regex"));

    /// Detects `panic!()` macro usage.
    pub fn panic_detector() -> AtomicSignal {
        AtomicSignal::new("panic", |content| PANIC_RE.is_match(content))
    }

    /// Detects `.unwrap()` calls.
    pub fn unwrap_detector() -> AtomicSignal {
        AtomicSignal::new("unwrap", |content| UNWRAP_RE.is_match(content))
    }

    /// Detects `.expect()` calls.
    pub fn expect_detector() -> AtomicSignal {
        AtomicSignal::new("expect", |content| EXPECT_RE.is_match(content))
    }

    /// Detects `unsafe {}` blocks.
    pub fn unsafe_detector() -> AtomicSignal {
        AtomicSignal::new("unsafe", |content| UNSAFE_RE.is_match(content))
    }

    /// Detects `mem::transmute` usage.
    pub fn transmute_detector() -> AtomicSignal {
        AtomicSignal::new("transmute", |content| TRANSMUTE_RE.is_match(content))
    }

    /// Detects TODO/FIXME/XXX markers.
    pub fn todo_detector() -> AtomicSignal {
        AtomicSignal::new("todo", |content| TODO_RE.is_match(content))
    }

    /// Detects test context (#[test] or #[cfg(test)]).
    pub fn test_context() -> AtomicSignal {
        AtomicSignal::new("test_context", |content| {
            content.contains("#[test]") || content.contains("#[cfg(test)]")
        })
    }

    /// Detects main function (exception context).
    pub fn main_function() -> AtomicSignal {
        AtomicSignal::new("main_fn", |content| content.contains("fn main()"))
    }
}

// ============================================================================
// STANDARD COMPOSITIONS
// ============================================================================

/// Pre-built signal expressions for common hook patterns.
pub mod compositions {
    use super::{SignalExpr, signals};

    /// Safety violation: panic OR unwrap OR expect (outside test context)
    pub fn safety_violation() -> SignalExpr {
        SignalExpr::and(vec![
            SignalExpr::or(vec![
                SignalExpr::atom(signals::panic_detector()),
                SignalExpr::atom(signals::unwrap_detector()),
                SignalExpr::atom(signals::expect_detector()),
            ]),
            SignalExpr::not(SignalExpr::atom(signals::test_context())),
        ])
    }

    /// Unsafe code: unsafe block OR transmute
    pub fn unsafe_code() -> SignalExpr {
        SignalExpr::or(vec![
            SignalExpr::atom(signals::unsafe_detector()),
            SignalExpr::atom(signals::transmute_detector()),
        ])
    }

    /// Clean code: NOT(safety_violation) AND NOT(unsafe_code) AND NOT(todo)
    pub fn clean_code() -> SignalExpr {
        SignalExpr::and(vec![
            SignalExpr::not(safety_violation()),
            SignalExpr::not(unsafe_code()),
            SignalExpr::not(SignalExpr::atom(signals::todo_detector())),
        ])
    }

    /// Test file: has test context marker
    pub fn is_test_file() -> SignalExpr {
        SignalExpr::atom(signals::test_context())
    }

    /// Main file exception: allows unwrap in main()
    pub fn main_exception() -> SignalExpr {
        SignalExpr::atom(signals::main_function())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_signal_basic() {
        let signal = AtomicSignal::new("contains_foo", |s| s.contains("foo"));
        assert!(signal.evaluate("hello foo world"));
        assert!(!signal.evaluate("hello bar world"));
    }

    #[test]
    fn test_atomic_always_never() {
        let always = AtomicSignal::always("always");
        let never = AtomicSignal::never("never");
        assert!(always.evaluate("anything"));
        assert!(!never.evaluate("anything"));
    }

    #[test]
    fn test_signal_expr_and() {
        let a = AtomicSignal::new("has_a", |s| s.contains('a'));
        let b = AtomicSignal::new("has_b", |s| s.contains('b'));
        let expr = SignalExpr::and(vec![SignalExpr::atom(a), SignalExpr::atom(b)]);

        assert!(expr.evaluate("ab"));
        assert!(!expr.evaluate("a"));
        assert!(!expr.evaluate("b"));
        assert!(!expr.evaluate("c"));
    }

    #[test]
    fn test_signal_expr_or() {
        let a = AtomicSignal::new("has_a", |s| s.contains('a'));
        let b = AtomicSignal::new("has_b", |s| s.contains('b'));
        let expr = SignalExpr::or(vec![SignalExpr::atom(a), SignalExpr::atom(b)]);

        assert!(expr.evaluate("ab"));
        assert!(expr.evaluate("a"));
        assert!(expr.evaluate("b"));
        assert!(!expr.evaluate("c"));
    }

    #[test]
    fn test_signal_expr_not() {
        let a = AtomicSignal::new("has_a", |s| s.contains('a'));
        let expr = SignalExpr::not(SignalExpr::atom(a));

        assert!(!expr.evaluate("a"));
        assert!(expr.evaluate("b"));
    }

    #[test]
    fn test_nested_expression() {
        // (A AND B) OR C
        let a = AtomicSignal::new("has_a", |s| s.contains('a'));
        let b = AtomicSignal::new("has_b", |s| s.contains('b'));
        let c = AtomicSignal::new("has_c", |s| s.contains('c'));

        let expr = SignalExpr::or(vec![
            SignalExpr::and(vec![SignalExpr::atom(a), SignalExpr::atom(b)]),
            SignalExpr::atom(c),
        ]);

        assert!(expr.evaluate("ab")); // A AND B
        assert!(expr.evaluate("c")); // C
        assert!(expr.evaluate("abc")); // Both
        assert!(!expr.evaluate("a")); // Only A
        assert!(!expr.evaluate("d")); // Neither
    }

    #[test]
    fn test_panic_detector() {
        let signal = signals::panic_detector();
        assert!(signal.evaluate("panic!(\"error\")"));
        assert!(signal.evaluate("  panic!  (  \"spaced\"  )"));
        assert!(!signal.evaluate("// panic! commented"));
    }

    #[test]
    fn test_unwrap_detector() {
        let signal = signals::unwrap_detector();
        assert!(signal.evaluate("result.unwrap()"));
        assert!(!signal.evaluate("result.unwrap_or(default)"));
    }

    #[test]
    fn test_safety_violation_composition() {
        let expr = compositions::safety_violation();

        // Panic outside test = violation
        assert!(expr.evaluate("fn foo() { panic!(\"boom\"); }"));

        // Panic inside test = NOT violation
        assert!(!expr.evaluate("#[test] fn foo() { panic!(\"ok\"); }"));

        // Clean code = NOT violation
        assert!(!expr.evaluate("fn foo() { Ok(()) }"));
    }

    #[test]
    fn test_evaluate_tracked() {
        let expr = SignalExpr::or(vec![
            SignalExpr::atom(signals::panic_detector()),
            SignalExpr::atom(signals::unwrap_detector()),
        ]);

        let result = evaluate_tracked(&expr, "x.unwrap()");
        assert!(result.fired);
        assert_eq!(result.fired_signals, vec!["unwrap"]);
        assert!(result.atoms_evaluated >= 1);
    }

    #[test]
    fn test_atom_count() {
        let expr = compositions::safety_violation();
        assert!(expr.atom_count() >= 4); // panic, unwrap, expect, test_context
    }

    #[test]
    fn test_signal_names() {
        let expr = SignalExpr::and(vec![
            SignalExpr::atom(signals::panic_detector()),
            SignalExpr::atom(signals::unwrap_detector()),
        ]);
        let names = expr.signal_names();
        assert!(names.contains(&"panic"));
        assert!(names.contains(&"unwrap"));
    }
}
