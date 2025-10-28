from pathlib import Path
import shutil
import json


overwrite_best = False
overall_best_filepath = Path("best_design/mydesign_yosys.v")

if not overall_best_filepath.exists():
    Path("best_design").mkdir(parents=True, exist_ok=True)
    overwrite_best = True
    current_nb_transistors_report = json.load(open(Path("synth/reports/mydesign_area_logic.rpt.json"), "r"))
    current_nb_transistors = int(
        "".join(
            filter(
                str.isdigit, current_nb_transistors_report["modules"]["\\mydesign_comb"]["estimated_num_transistors"]
            )
        )
    )

else:
    best_counts_dict = json.load(open(Path("best_nb_transistors.json"), "r"))
    best_nb_transistors = best_counts_dict["nb_transistors"]
    print("best", best_nb_transistors)

    current_nb_transistors_report = json.load(open(Path("synth/reports/mydesign_area_logic.rpt.json"), "r"))
    current_nb_transistors = int(
        "".join(
            filter(
                str.isdigit, current_nb_transistors_report["modules"]["\\mydesign_comb"]["estimated_num_transistors"]
            )
        )
    )
    print("current", current_nb_transistors)

    if current_nb_transistors < best_nb_transistors:
        overwrite_best = True

if overwrite_best:
    best_dict = {"nb_transistors": current_nb_transistors}
    json.dump(best_dict, open(Path("best_nb_transistors.json"), "w"))

    files_to_copy = [
        Path("synth/reports/mydesign_area_logic.rpt.json"),
        Path("synth/reports/mydesign_area_logic.rpt"),
        Path("synth/reports/mydesign_synth.rpt"),
        Path("synth/out/mydesign_yosys.v"),
    ]
    for filepath in files_to_copy:
        shutil.copy(filepath, Path("best_design") / filepath.name)
    print("Best design overwritten.")
else:
    print("Skipped overwritting.")
