/*
 * solver.c — Fast Sudoku solver with constraint-propagation trace output.
 *
 * Reads 81-char puzzle strings (one per line) from stdin.
 * Writes binary records to stdout:
 *   uint8_t  status        (1=solved, 0=failed)
 *   char     solution[81]  (ASCII '1'-'9')
 *   uint8_t  trace_len
 *   uint8_t  trace[trace_len][3]  (row, col, digit)
 *
 * Usage: ./solver [--seed N] < puzzles.txt > traces.bin
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALL_BITS 0x1FF /* bits 0-8 set = digits 1-9 */

/* ---- Precomputed tables ---- */

static int peers[81][20];
static int units[81][3][9];

static void init_tables(void) {
    /* Build all 27 units */
    int unitlist[27][9];
    int u = 0;
    /* Columns */
    for (int c = 0; c < 9; c++, u++)
        for (int r = 0; r < 9; r++)
            unitlist[u][r] = r * 9 + c;
    /* Rows */
    for (int r = 0; r < 9; r++, u++)
        for (int c = 0; c < 9; c++)
            unitlist[u][c] = r * 9 + c;
    /* Boxes */
    for (int br = 0; br < 3; br++)
        for (int bc = 0; bc < 3; bc++, u++) {
            int k = 0;
            for (int dr = 0; dr < 3; dr++)
                for (int dc = 0; dc < 3; dc++)
                    unitlist[u][k++] = (br*3+dr)*9 + (bc*3+dc);
        }

    for (int i = 0; i < 81; i++) {
        /* Find the 3 units containing cell i */
        int ui = 0;
        for (int u = 0; u < 27; u++) {
            int found = 0;
            for (int k = 0; k < 9; k++)
                if (unitlist[u][k] == i) { found = 1; break; }
            if (found) {
                memcpy(units[i][ui], unitlist[u], 9 * sizeof(int));
                ui++;
            }
        }

        /* Build peers: union of unit members minus self */
        int seen[81] = {0};
        seen[i] = 1;
        int pi = 0;
        for (int u = 0; u < 3; u++)
            for (int k = 0; k < 9; k++) {
                int c = units[i][u][k];
                if (!seen[c]) { seen[c] = 1; peers[i][pi++] = c; }
            }
    }
}

/* ---- Fisher-Yates shuffle ---- */

static void shuffle_int(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/* Shuffle a row of 9 ints as a unit (swap entire rows) */
static void shuffle_unit_rows(int units_cell[3][9]) {
    for (int i = 2; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp[9];
        memcpy(tmp, units_cell[i], 9 * sizeof(int));
        memcpy(units_cell[i], units_cell[j], 9 * sizeof(int));
        memcpy(units_cell[j], tmp, 9 * sizeof(int));
    }
}

/* ---- Solver state ---- */

/* Per-puzzle shuffled tables (shared across search tree, not copied) */
typedef struct {
    int peers[81][20];
    int units[81][3][9];
    int elim_order[9];       /* shuffled digit elimination order (bit positions) */
} ShuffledTables;

typedef struct {
    uint16_t values[81];
    uint8_t  clues[81];    /* 1 if cell is a clue */
    uint8_t  trace[81][3]; /* (row, col, digit) */
    int      trace_len;
    ShuffledTables *tables;
} State;

static void init_shuffled_tables(ShuffledTables *t) {
    memcpy(t->peers, peers, sizeof(peers));
    memcpy(t->units, units, sizeof(units));
    for (int i = 0; i < 81; i++) {
        shuffle_int(t->peers[i], 20);
        for (int u = 0; u < 3; u++)
            shuffle_int(t->units[i][u], 9);
        shuffle_unit_rows(t->units[i]);
    }
    /* Shuffle digit elimination order (bit positions 0-8) */
    for (int i = 0; i < 9; i++) t->elim_order[i] = i;
    shuffle_int(t->elim_order, 9);
}

static int eliminate(State *s, int cell, uint16_t d_bit);

static int assign(State *s, int cell, uint16_t d_bit) {
    uint16_t other = s->values[cell] & ~d_bit;
    for (int i = 0; i < 9; i++) {
        uint16_t bit = 1 << s->tables->elim_order[i];
        if (other & bit) {
            if (!eliminate(s, cell, bit))
                return 0;
        }
    }
    return 1;
}

static int eliminate(State *s, int cell, uint16_t d_bit) {
    if (!(s->values[cell] & d_bit))
        return 1; /* already eliminated */
    uint16_t new_val = s->values[cell] & ~d_bit;
    if (new_val == 0)
        return 0;
    s->values[cell] = new_val;

    /* Naked single */
    if (!(new_val & (new_val - 1))) {
        int digit = __builtin_ctz(new_val) + 1;
        if (!s->clues[cell]) {
            int r = cell / 9, c = cell % 9;
            s->trace[s->trace_len][0] = (uint8_t)r;
            s->trace[s->trace_len][1] = (uint8_t)c;
            s->trace[s->trace_len][2] = (uint8_t)digit;
            s->trace_len++;
        }
        for (int p = 0; p < 20; p++) {
            if (!eliminate(s, s->tables->peers[cell][p], new_val))
                return 0;
        }
    }

    /* Hidden single */
    for (int u = 0; u < 3; u++) {
        int count = 0, place = -1;
        for (int k = 0; k < 9; k++) {
            int sq = s->tables->units[cell][u][k];
            if (s->values[sq] & d_bit) {
                count++;
                if (count > 1) break;
                place = sq;
            }
        }
        if (count == 0) return 0;
        if (count == 1) {
            if (!assign(s, place, d_bit))
                return 0;
        }
    }
    return 1;
}

/* Count solutions up to limit; returns as soon as limit is reached. */
static int count_solutions(State *s, int limit) {
    int min_count = 10;
    int best = -1;
    for (int i = 0; i < 81; i++) {
        if (s->values[i] == 0) return 0; /* contradiction */
        int cnt = __builtin_popcount(s->values[i]);
        if (cnt > 1 && cnt < min_count) {
            min_count = cnt;
            best = i;
        }
    }
    if (best == -1) return 1; /* all cells determined */

    int total = 0;
    uint16_t bits = s->values[best];
    for (int d = 0; d < 9; d++) {
        uint16_t bit = 1 << d;
        if (!(bits & bit)) continue;
        State copy = *s;
        if (assign(&copy, best, bit)) {
            total += count_solutions(&copy, limit - total);
            if (total >= limit) return total;
        }
    }
    return total;
}

static int search(State *s) {
    /* MRV with random tie-breaking */
    int min_count = 10;
    int tied[81], n_tied = 0;

    for (int i = 0; i < 81; i++) {
        if (s->values[i] == 0) return 0;
        int cnt = __builtin_popcount(s->values[i]);
        if (cnt > 1) {
            if (cnt < min_count) {
                min_count = cnt;
                tied[0] = i;
                n_tied = 1;
            } else if (cnt == min_count) {
                tied[n_tied++] = i;
            }
        }
    }
    if (n_tied == 0) return 1; /* solved */

    int cell = tied[rand() % n_tied];
    uint16_t bits = s->values[cell];

    for (int i = 0; i < 9; i++) {
        uint16_t bit = 1 << s->tables->elim_order[i];
        if (!(bits & bit)) continue;
        State copy = *s;
        if (assign(&copy, cell, bit)) {
            if (search(&copy)) {
                *s = copy;
                return 1;
            }
        }
    }
    return 0;
}

/*
 * Returns: 0 = no solution, 1 = solved (unique if check_unique=0),
 *          2 = multiple solutions (only when check_unique=1).
 */
static int solve(const char *puzzle, char *solution, uint8_t trace_out[][3], int *trace_len,
                 int check_unique) {
    State s;
    for (int i = 0; i < 81; i++)
        s.values[i] = ALL_BITS;
    memset(s.clues, 0, sizeof(s.clues));
    s.trace_len = 0;

    /* Shuffle peer/unit tables for this puzzle */
    ShuffledTables tables;
    s.tables = &tables;
    init_shuffled_tables(&tables);

    /* Collect clue indices and shuffle */
    int clue_idx[81];
    int n_clues = 0;
    for (int i = 0; i < 81; i++) {
        if (puzzle[i] >= '1' && puzzle[i] <= '9')
            clue_idx[n_clues++] = i;
    }
    shuffle_int(clue_idx, n_clues);

    /* Parse clues in shuffled order */
    for (int k = 0; k < n_clues; k++) {
        int i = clue_idx[k];
        s.clues[i] = 1;
        if (!assign(&s, i, 1 << (puzzle[i] - '1')))
            return 0;
    }

    /* Save state after constraint propagation for uniqueness check */
    State s_after_clues = s;

    /* Check if solved */
    int solved = 1;
    for (int i = 0; i < 81; i++) {
        if (__builtin_popcount(s.values[i]) != 1) { solved = 0; break; }
    }

    if (!solved) {
        if (!search(&s))
            return 0;
    }

    /* Extract solution */
    for (int i = 0; i < 81; i++)
        solution[i] = '1' + __builtin_ctz(s.values[i]);

    memcpy(trace_out, s.trace, s.trace_len * 3);
    *trace_len = s.trace_len;

    /* Uniqueness check: look for a second solution */
    if (check_unique && count_solutions(&s_after_clues, 2) >= 2)
        return 2;

    return 1;
}

int main(int argc, char **argv) {
    unsigned seed = 42;
    int check_unique = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--check-unique") == 0)
            check_unique = 1;
    }
    srand(seed);
    init_tables();

    char line[256];
    char solution[81];
    uint8_t trace[81][3];
    int trace_len;
    long count = 0;
    int progress_interval = 100000;

    while (fgets(line, sizeof(line), stdin)) {
        /* Skip lines that don't look like puzzles */
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len < 81) continue;
        /* Check first char is digit or dot */
        if (!(line[0] >= '0' && line[0] <= '9') && line[0] != '.') continue;

        int ok = solve(line, solution, trace, &trace_len, check_unique);
        uint8_t status = (uint8_t)ok;
        fwrite(&status, 1, 1, stdout);
        fwrite(solution, 1, 81, stdout);
        uint8_t tl = (uint8_t)trace_len;
        fwrite(&tl, 1, 1, stdout);
        if (trace_len > 0)
            fwrite(trace, 3, (size_t)trace_len, stdout);

        count++;
        if (count % progress_interval == 0)
            fprintf(stderr, "\r  solved %ld puzzles...", count);
    }
    if (count >= progress_interval)
        fprintf(stderr, "\r  solved %ld puzzles (done)\n", count);

    return 0;
}
