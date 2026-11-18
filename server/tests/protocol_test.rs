//! `tensorquill-server protocol`: the one number the Python SDK checks before it runs the binary.

use std::process::Command;

#[test]
fn protocol_prints_one_whole_number_and_needs_no_store() {
    let output = Command::new(env!("CARGO_BIN_EXE_tensorquill-server"))
        .arg("protocol")
        .output()
        .expect("the binary runs");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.stderr.is_empty(), "{}", String::from_utf8_lossy(&output.stderr));
    let printed = String::from_utf8(output.stdout).expect("text");
    // One line, digits only: the SDK reads it with `int()` and refuses anything else.
    assert!(
        printed.ends_with('\n') && printed.trim_end().chars().all(|c| c.is_ascii_digit()),
        "{printed:?}"
    );
    assert!(printed.trim_end().parse::<u32>().expect("a whole number") >= 1);
}
