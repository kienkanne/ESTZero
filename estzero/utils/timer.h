#ifndef TIMER_H
#define TIMER_H

#include <sys/resource.h>

#include <chrono>
#include <print>
#include <string>
#include <vector>

namespace timing {

// Peak resident set size (high-water mark of physical RAM used by this process, heap included) in KB
inline long peak_rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

struct Record {
    std::string name;
    double seconds;
    long calls;
    bool last;
};

inline std::vector<Record>& registry() {
    // local static, instantiate once and live for program's lifetime
    static std::vector<Record> r;
    return r;
}

inline void add(const std::string& name, double seconds, bool last) {
    for (auto& rec : registry()) {
        if (rec.name == name) {
            rec.seconds += seconds;
            rec.calls += 1;
            return;
        }
    }
    registry().push_back({name, seconds, 1, last});
}

class Scoped {
private:
    std::string name_;
    std::chrono::steady_clock::time_point t0_;
    bool last_;

public:
    explicit Scoped(std::string name, bool last = false)
        : name_(std::move(name)), t0_(std::chrono::steady_clock::now()), last_(std::move(last)) {}

    ~Scoped() {
        const double s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        add(name_, s, last_);
    }

    Scoped(const Scoped&) = delete;
    Scoped& operator=(const Scoped&) = delete;
};

inline void report() {
    std::println("\n{:<26} {:>12} {:>8} {:>12}", "Phase", "Time (s)", "Calls",
                 "Avg (ms)");
    std::println("{:=<62}", "");
    for (const auto& r : registry()) {
        const double avg_ms = r.calls > 0 ? 1e3 * r.seconds / r.calls : 0.0;
        if (r.last) {
            std::println("{:-<62}", "");
        }
        std::println("{:<26} {:>12.4f} {:>8} {:>12.3f}", r.name, r.seconds, r.calls, avg_ms);
    }
    std::println("{:=<62}", "");
    std::println("{} {:.1f} MB", "peak RSS:", peak_rss_kb() / 1024.0);
}

}

#endif
