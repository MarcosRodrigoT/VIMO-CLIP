import h5py
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple


def analyze_h5_structure(filepath: str) -> Dict[str, Any]:
    """
    Analyze the structure of an H5 file and return a dictionary describing its structure.
    """
    structure = {"groups": {}, "datasets": {}, "num_groups": 0, "num_datasets": 0, "sample_group_structure": None, "all_groups_same_structure": True, "group_dataset_info": []}

    def visit_item(name, obj):
        if isinstance(obj, h5py.Group):
            structure["num_groups"] += 1
            group_info = {"name": name, "datasets": {}, "subgroups": []}

            # Analyze datasets within this group
            for key in obj.keys():
                item = obj[key]
                if isinstance(item, h5py.Dataset):
                    dataset_info = {"name": key, "shape": item.shape, "dtype": str(item.dtype), "size": item.size}
                    group_info["datasets"][key] = dataset_info
                elif isinstance(item, h5py.Group):
                    group_info["subgroups"].append(key)

            structure["groups"][name] = group_info
            structure["group_dataset_info"].append(group_info)

        elif isinstance(obj, h5py.Dataset):
            structure["num_datasets"] += 1
            dataset_info = {"name": name, "shape": obj.shape, "dtype": str(obj.dtype), "size": obj.size}
            structure["datasets"][name] = dataset_info

    with h5py.File(filepath, "r") as f:
        f.visititems(visit_item)

    # Analyze if all groups have the same structure
    if structure["group_dataset_info"]:
        first_group = structure["group_dataset_info"][0]
        sample_structure = {
            "dataset_names": list(first_group["datasets"].keys()),
            "dataset_dtypes": {k: v["dtype"] for k, v in first_group["datasets"].items()},
            "num_datasets": len(first_group["datasets"]),
            "has_subgroups": len(first_group["subgroups"]) > 0,
            "subgroup_names": first_group["subgroups"],
        }
        structure["sample_group_structure"] = sample_structure

        # Check if all groups follow the same structure
        for group_info in structure["group_dataset_info"]:
            current_structure = {
                "dataset_names": list(group_info["datasets"].keys()),
                "dataset_dtypes": {k: v["dtype"] for k, v in group_info["datasets"].items()},
                "num_datasets": len(group_info["datasets"]),
                "has_subgroups": len(group_info["subgroups"]) > 0,
                "subgroup_names": group_info["subgroups"],
            }

            if current_structure != sample_structure:
                structure["all_groups_same_structure"] = False
                break

    return structure


def print_structure_summary(structure: Dict[str, Any], filename: str):
    """
    Print a human-readable summary of the H5 file structure.
    """
    print(f"\n=== Structure Summary for {filename} ===")
    print(f"Total groups: {structure['num_groups']}")
    print(f"Total root-level datasets: {structure['num_datasets']}")
    print(f"All groups have same structure: {structure['all_groups_same_structure']}")

    if structure["sample_group_structure"]:
        print(f"\nGroup structure pattern:")
        sample = structure["sample_group_structure"]
        print(f"  - Datasets per group: {sample['num_datasets']}")
        print(f"  - Dataset names: {sample['dataset_names']}")
        print(f"  - Dataset dtypes: {sample['dataset_dtypes']}")
        print(f"  - Has subgroups: {sample['has_subgroups']}")
        if sample["subgroup_names"]:
            print(f"  - Subgroup names: {sample['subgroup_names']}")

    # Show some example groups and their dataset shapes
    print(f"\nSample groups and dataset shapes:")
    for i, group_info in enumerate(structure["group_dataset_info"][:5]):  # Show first 5 groups
        print(f"  Group '{group_info['name']}':")
        for dataset_name, dataset_info in group_info["datasets"].items():
            print(f"    - {dataset_name}: shape={dataset_info['shape']}, dtype={dataset_info['dtype']}")
        if i >= 4 and len(structure["group_dataset_info"]) > 5:
            print(f"  ... and {len(structure['group_dataset_info']) - 5} more groups")
            break


def compare_structures(struct1: Dict[str, Any], struct2: Dict[str, Any], file1: str, file2: str) -> bool:
    """
    Compare two H5 file structures and return True if they match.
    """
    print(f"\n=== Comparing Structures ===")

    issues = []

    # Compare basic counts
    if struct1["num_datasets"] != struct2["num_datasets"]:
        issues.append(f"Different number of root-level datasets: {file1} has {struct1['num_datasets']}, {file2} has {struct2['num_datasets']}")

    # Compare group structure consistency
    if struct1["all_groups_same_structure"] != struct2["all_groups_same_structure"]:
        issues.append(f"Group structure consistency differs: {file1}={struct1['all_groups_same_structure']}, {file2}={struct2['all_groups_same_structure']}")

    # Compare sample group structures
    if struct1["sample_group_structure"] and struct2["sample_group_structure"]:
        s1 = struct1["sample_group_structure"]
        s2 = struct2["sample_group_structure"]

        if s1["dataset_names"] != s2["dataset_names"]:
            issues.append(f"Different dataset names in groups: {file1} has {s1['dataset_names']}, {file2} has {s2['dataset_names']}")

        if s1["dataset_dtypes"] != s2["dataset_dtypes"]:
            issues.append(f"Different dataset dtypes in groups: {file1} has {s1['dataset_dtypes']}, {file2} has {s2['dataset_dtypes']}")

        if s1["num_datasets"] != s2["num_datasets"]:
            issues.append(f"Different number of datasets per group: {file1} has {s1['num_datasets']}, {file2} has {s2['num_datasets']}")

        if s1["has_subgroups"] != s2["has_subgroups"]:
            issues.append(f"Different subgroup presence: {file1} has subgroups={s1['has_subgroups']}, {file2} has subgroups={s2['has_subgroups']}")

        if s1["subgroup_names"] != s2["subgroup_names"]:
            issues.append(f"Different subgroup names: {file1} has {s1['subgroup_names']}, {file2} has {s2['subgroup_names']}")

    elif bool(struct1["sample_group_structure"]) != bool(struct2["sample_group_structure"]):
        issues.append(f"One file has groups while the other doesn't")

    # Print results
    if issues:
        print("❌ STRUCTURES DO NOT MATCH!")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ STRUCTURES MATCH!")
        print("Both H5 files follow the same structural pattern.")
        return True


def get_detailed_comparison(struct1: Dict[str, Any], struct2: Dict[str, Any], file1: str, file2: str):
    """
    Print detailed comparison information.
    """
    print(f"\n=== Detailed Comparison ===")

    print(f"\nFile sizes:")
    print(f"  {file1}: {struct1['num_groups']} groups")
    print(f"  {file2}: {struct2['num_groups']} groups")

    # Compare dataset shapes distribution
    shapes1 = []
    shapes2 = []

    for group_info in struct1["group_dataset_info"]:
        for dataset_info in group_info["datasets"].values():
            shapes1.append(dataset_info["shape"])

    for group_info in struct2["group_dataset_info"]:
        for dataset_info in group_info["datasets"].values():
            shapes2.append(dataset_info["shape"])

    if shapes1 and shapes2:
        print(f"\nDataset shape statistics:")
        print(f"  {file1}: {len(shapes1)} total datasets")
        print(f"    - Shape range: min={min(shapes1)}, max={max(shapes1)}")
        print(f"  {file2}: {len(shapes2)} total datasets")
        print(f"    - Shape range: min={min(shapes2)}, max={max(shapes2)}")


def main():
    parser = argparse.ArgumentParser(description="Compare the structure of two H5 files")
    parser.add_argument("file1", help="Path to first H5 file")
    parser.add_argument("file2", help="Path to second H5 file")
    parser.add_argument("--detailed", action="store_true", help="Show detailed comparison information")

    args = parser.parse_args()

    try:
        # Analyze both files
        print(f"Analyzing {args.file1}...")
        struct1 = analyze_h5_structure(args.file1)

        print(f"Analyzing {args.file2}...")
        struct2 = analyze_h5_structure(args.file2)

        # Print summaries
        print_structure_summary(struct1, args.file1)
        print_structure_summary(struct2, args.file2)

        # Compare structures
        match = compare_structures(struct1, struct2, args.file1, args.file2)

        # Show detailed comparison if requested
        if args.detailed:
            get_detailed_comparison(struct1, struct2, args.file1, args.file2)

        # Exit with appropriate code
        exit(0 if match else 1)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
