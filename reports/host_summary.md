# AutoDL host observation

Observed 2026-11-18T22:43:40.573936+08:00. NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 595.71.05, 97887 MiB device memory. At inventory the device used 0 MiB and reported 0 % utilization.

CPU quota 2500000 100000, affinity 208, effective budget 25 CPUs. Memory limit 120 GiB, initial cgroup charge 11.2 GiB. Project disk free at inventory 469.8 GiB.

The isolated Conda prefix and dedicated package, compilation and model caches are under .local/. One formal trainer has access to the whole device. Compilation uses Ninja with a live memory-based job budget. Frame validation uses one native thread per worker. Poisson explicitly receives the effective CPU budget.
