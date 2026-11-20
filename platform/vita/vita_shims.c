/* libc pieces newlib on the Vita doesn't have. */


/* ps2_memcard.c takes an exclusive lock on the memory card file so two copies
 * of the game can't share it. Only one app runs at a time on the Vita, so the
 * lock always succeeds. */
int flock(int fd, int op) {
    (void)fd; (void)op;
    return 0;
}
