import pandas as pd
from collections import defaultdict
import random
from copy import deepcopy
from collections import defaultdict


def get_image_to_class_count_dataframe(coco_object):
    """Returns a dataframe with shape (N_images,1+N_clases).
    
    The first column is the image ID and the remaining
    colums are the class counts."""
    # Step 2: Extract image IDs and category mappings
    categories = {cat['id']: cat['name'] for cat in coco_object['categories']}
    image_ids = [img['id'] for img in coco_object['images']]

    # Step 3: Count annotations per image per category
    image_class_counts = defaultdict(lambda: defaultdict(int))
    for ann in coco_object['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_class_counts[image_id][category_id] += 1

    # Step 4: Construct DataFrame
    rows = []
    for image_id in image_ids:
        row = {'image_id': image_id}
        for cat_id in categories:
            row[categories[cat_id]] = image_class_counts[image_id][cat_id]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def sub_sample_count_dataframe(df,n_images,n_classes,random_state=42):
    """
        Sub-samples a pandas dataframe returned 
        from get_image_to_class_count_dataframe
    """

    # 1. Remove the 'image_id' column to work only with class counts
    all_class_cols = [col for col in df.columns if col != 'image_id']
    # 2. Randomly select 10 classes
    selected_classes = random.sample(all_class_cols, n_classes)
    # 3. Filter full dataframe to only include rows where at least one of the selected classes has a non-zero count
    df_nonzero = df[df[selected_classes].sum(axis=1) > 0]
    # 4. Sample 200 images from that filtered set
    df_subsample = df_nonzero[['image_id'] + selected_classes].sample(n=n_images).reset_index(drop=True)
    # Show result
    df_subsample.head()
    return df_subsample


def get_dataframe_count_datastructures(df):
    """
    Construct auxiliary data structures from a DataFrame of image IDs and class counts.

    This function expects a pandas DataFrame where:
      - The first column is 'image_id' (unique identifiers for images).
      - The remaining columns correspond to class names, with integer values
        representing the counts of that class in the given image.

    It extracts mappings between image IDs, class names, and their integer indices,
    and builds the class count matrix suitable for downstream tasks.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame with:
          - 'image_id' column (unique identifiers for images).
          - One or more class columns (counts per class for each image).

    Returns
    -------
    class_counts_per_image : numpy.ndarray, shape (num_images, num_classes)
        Matrix of class counts, where entry [i, c] is the count of class `c` 
        in image `i`.
    index_to_image_id : dict[int, Any]
        Mapping from image index (row position) to its image ID.
    index_to_class_name : dict[int, str]
        Mapping from class index (column position) to class name.
    class_names : list[str]
        Ordered list of class names corresponding to the columns in the DataFrame.
    image_ids : list[Any]
        Ordered list of image IDs corresponding to the rows in the DataFrame.

    Notes
    -----
    - Image indices are assigned based on the row order of `df`.
    - Class indices are assigned based on the column order of `df`, 
      excluding 'image_id'.
    """
    # Assume df is your DataFrame with:
    # - 'image_id' as the first column
    # - class columns as the rest

    # Extract image ID and class column names
    image_ids = df['image_id'].tolist()
    class_names = [col for col in df.columns if col != 'image_id']

    # Build mappings
    image_id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    index_to_image_id = {idx: img_id for img_id, idx in image_id_to_index.items()}

    class_name_to_index = {cls: idx for idx, cls in enumerate(class_names)}
    index_to_class_name = {idx: cls for cls, idx in class_name_to_index.items()}

    # Build W[i][c] matrix (class counts c for image i)
    class_counts_per_image = df[class_names].to_numpy()  # shape: (num_images, num_classes)

    return class_counts_per_image,index_to_image_id,index_to_class_name,class_names,image_ids



def construct_coco_split_from_assignments(original_coco_object, assignments,
                                          index_to_image_id,
                                          assignment_index_to_name):
    """
    Construct COCO-format dataset splits based on a set of image-to-assignment mappings.

    Given an original COCO-style dictionary and a mapping of images to assignment
    indices, this function partitions the dataset into multiple new COCO objects,
    one per assignment class. Each new object contains the subset of images and
    annotations belonging to that assignment class, while reusing the original
    metadata (info, licenses, and categories).

    Parameters
    ----------
    original_coco_object : dict
        A dictionary in COCO format containing the fields:
        - "info": dataset-level metadata
        - "licenses": license information
        - "categories": list of category definitions
        - "images": list of image records, each with an "id"
        - "annotations": list of annotation records, each with an "image_id"
    assignments : dict[int, int]
        Mapping from DataFrame row index (image index) to an assignment index.
    index_to_image_id : dict[int, Any]
        Mapping from image index (row position in the DataFrame) to actual COCO image ID.
    assignment_index_to_name : dict[int, str]
        Mapping from assignment index to a human-readable class/partition name.

    Returns
    -------
    coco_per_assignment : dict[str, dict]
        Dictionary where keys are assignment names (from `assignment_index_to_name`)
        and values are new COCO-format objects containing only the images and
        annotations belonging to that assignment.

    """

    # Step 2: Reuse mappings from earlier step
    # index_to_image_id: maps i (DataFrame row index) to actual image_id

    # Step 3: Build reverse mapping: assignment_class → set of image_ids
    class_to_image_ids = defaultdict(set)
    for i, assigned_class in assignments.items():
        image_id = index_to_image_id[i]
        class_to_image_ids[assigned_class].add(image_id)

    # Step 4: Build COCO objects per assignment class
    coco_per_assignment = {}

    for assignment_idx, image_ids in class_to_image_ids.items():
        name = assignment_index_to_name[assignment_idx]

        # Build new COCO object
        new_coco = {
            "info": deepcopy(original_coco_object.get("info", {})),
            "licenses": deepcopy(original_coco_object.get("licenses", [])),
            "categories": deepcopy(original_coco_object["categories"]),
            "images": [img for img in original_coco_object["images"] if img["id"] in image_ids],
            "annotations": [ann for ann in original_coco_object["annotations"] if ann["image_id"] in image_ids],
        }

        coco_per_assignment[name] = new_coco

    return coco_per_assignment
