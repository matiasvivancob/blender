Minimal repro for the clang gather finding (see ../README.md, "Compiler findings"):

  clang++ -O3 -march=haswell -ffp-contract=on -S dense_rm_gather_repro.cpp -o - | grep -c vgather   # 0
  clang++ -O3 -march=skylake -ffp-contract=on -S dense_rm_gather_repro.cpp -o - | grep -c vgather   # >100

Same source, same ISA level for the hot loop; only the tuning model differs.
