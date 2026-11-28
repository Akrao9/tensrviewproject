//! The lock rule (SV10 a, `WAVE_AN_CONTRACT.md` section 11): every standard mutex of `ingest.rs`
//! and `watchdog.rs` is taken through [`locked`], so a panic while one is held does not end every
//! later request that takes it. Tokio's mutexes do not poison and are not taken through it.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// Takes `mutex`, whether or not a panic while it was held poisoned it.
///
/// A poisoned mutex is taken as it is: what the panic left half changed stays as it was, which the
/// maps behind these mutexes tolerate (a rank's entry is changed again at its next event, and a
/// status or report is written again at the next pass). The poison mark is not cleared, so
/// `is_poisoned` still reads true after.
pub(crate) fn locked<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::locked;
    use std::sync::Mutex;

    /// A panic between two changes made under the lock: the next taker gets the guard, sees the
    /// first change and not the second, and can change the value again.
    #[test]
    fn a_poisoned_mutex_is_taken_with_what_the_panic_left() {
        let mutex = Mutex::new(vec![1]);
        std::thread::scope(|scope| {
            let holder = scope.spawn(|| {
                let mut held = locked(&mutex);
                held.push(2);
                panic!("a panic while the lock was held, from a lane AN-T test");
            });
            assert!(holder.join().is_err());
        });
        assert!(mutex.is_poisoned());
        assert_eq!(*locked(&mutex), [1, 2]);
        locked(&mutex).push(3);
        assert_eq!(*locked(&mutex), [1, 2, 3]);
        assert!(mutex.is_poisoned(), "the mark is not cleared");
    }

    #[test]
    fn a_mutex_not_poisoned_is_taken_as_lock_takes_it() {
        let mutex = Mutex::new(5);
        *locked(&mutex) += 1;
        assert_eq!(*locked(&mutex), 6);
        assert!(!mutex.is_poisoned());
    }
}
