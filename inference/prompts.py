# Prompt templates for 3D visual grounding tasks

target_name_select_user_prompt = """You will analyze a given query sentence to determine the main target object being described. 
Then, from the provided list of object names, you will select the best-matching name that corresponds to the target object.
Follow these steps:
1.Identify the target object in the query sentence based on its description.
2.Compare the identified target object with the available options in the provided list.
3.Select the best-matching name from the list.
4.Output the result as a JSON file in the format: {"target_name": "<name>"}
"""


target_name_select_sys_prompt = """You are working on a 3D visual grounding task, which involves receiving a query that specifies a particular object by describing its attributes and grounding conditions to uniquely identify the object. 
Now, I need you to first parse this query, return the category of the object to be found. 
Sometimes the object's category is not explicitly specified, and you need to deduce it through reasoning. 
If you cannot deduce after reasoning, you can use 'unknown' for the category.
Your response should be formatted in JSON

Here are some examples:

Input:
Query: this is a brown cabinet. it is to the right of a picture.

Output:
{
    "target_class": "cabinet"
}

Input:
Query: it is a wooden computer desk. the desk is in the sleeping area, across from the living room. the desk is in the corner of the room, between the nightstand and where the shelf and window are.

Output:
{
    "target_class": "desk"
}

Input:
Query: In the room is a set of desks along a wall with windows totaling 4 desks. Opposite this wall is another wall with a door and two desks. The desk of interest is the closest desk to the door. This desk has nothing on it, no monitor, etc.

Output:
{
    "target_class": "desk"
}
"""


view_selection_sys_prompt = """You are good at finding the object in a 3D scene based on a given query description."""
view_selection_user_prompt = """These images show different views of a room. You need to find the {target_class} in this query description: \"{text}\". 
Please review all view images to find the target object and select the view that you can see the target object most clearly. 
Output your answer in JSON format with these keys: 
"reasoning": Explain how you identified the target object, and why you choose this view.
"view": The number of the view is in the top left corner of the corresponding image. e.g., "view": "2"
"""


# top-k
topk_id_user_prompt = """Here is the annotated image of the selected view. 
All objects belonging to the {target_class} class is labeled by a unique number (ID) in red color on it.
Please select the object ID that best matches the given query description: \"{text}\".
Carefully analyze the specified conditions (such as shape, color, relative position with surrounding objects) in the given query, then select top-{n_topk} best-matched object IDs.
The selected top-{n_topk} object IDs should be sorted in descending order of confidence.
The object ID should be chosen from this list {object_id_list}
Output your answer in JSON format with these keys: 
"reasoning": Explain how you identified and ranked the top-{n_topk} target object IDs.
"object_id": A list of{n_topk} selected target object IDs. e.g., [1, 2, 3, 4, 5]
"""


# top-k with crop object grid image
topk_id_crop_user_prompt = """Please select the object ID that best matches the given query description: \"{text}\".
Here are two images. 
In the global view image, the detected objects are annotated in different object IDs. 
And another image shows all detected objects cropped from 2D images. The same object is annotated in the same ID in both images.
Carefully analyze the query text to identify key attributes of the target object. 
Then review the annotated 3D scene view and the cropped object images to select {n_topk} object IDs that best matches the given query description.
The selected {n_topk} object IDs should be sorted in descending order of confidence.
Output your answer in JSON format with these keys: 
"reasoning": Explain how you identified and ranked the top {n_topk} target object IDs.
"object_id": {n_topk} selected target object IDs. e.g., [1, 2, 3, 4, 5]
"""


object_id_user_prompt = """
You are given a set of images depicting an indoor scene:
One global view image that shows a 3D layout of the room from a fixed perspective.
Several camera images taken from different viewpoints around the room.
Objects in the scene are labeled with unique IDs (in red) on both the global and camera images.

Your task is to identify the object ID that best matches a given query description.
Use the following step-by-step approach:

1. Start with the global view image:
Understand the room's layout and spatial organization.
Identify the candidate objects that may match the query based on their position and label.

2. Verify using the camera images:
Cross-check the candidate object(s) from the global view against the camera images.
Look for visual characteristics (e.g., color, shape, size, material) and relative positioning to confirm the match.
Note: Some annotations (IDs) may be slightly offset from the object's actual center.
Consider the full spatial footprint of the object you identified in both global and camera views.
You should choose the ID that most accurately represents the center or main body of the correct object.

3. If no object fully meets the query, go back to the global view and reassess alternative candidates.
Repeat the above process until you find the best match.

Task:
Select the object ID of {target_class} that best matches the following query description:
\"{text}\"

Object IDs to choose from: {object_id_list}

Output your answer in JSON format with these keys: 
"reasoning": Explain how you compare and analyze all given images and select one best-matched target object ID.
"object_id": The selected best-matched target object ID from given list.
"""


object_id_user_prompt_relation = """
You are provided with a set of images depicting an indoor scene:
- A **global view image** showing the room's 3D layout from a fixed perspective.
- Several **camera images** captured from different viewpoints around the room.

All objects of interest in the scene are labeled with unique **object IDs** (in red), which are consistent across both the global and camera images.

Your task is to identify the object ID that best matches the given query description. Follow the steps below:

---

1. **Start with the global view image**:
   - Analyze the overall spatial layout and object distribution in the room.
   - Use the global view to evaluate **view-independent spatial relationships**, which do not rely on a specific viewpoint:
     Examples include:
     - `near`, `close to`, `next to`, `far`, `above`, `below`, `under`, `on`, `top of`, `middle`, `opposite`

2. **Then examine the camera images**:
   - Validate candidate objects identified from the global view.
   - Evaluate **visual features**: color, shape, size, texture, and material.
   - Use camera views to judge **view-dependent spatial relationships**, which depend on the camera perspective:
     Examples include:
     - `left`, `right`, `in front of`, `behind`, `back`, `facing`, `looking`, `between`, `across from`, `leftmost`, `rightmost`

    Tip: Annotations may not always be at the center of the object. Focus on the full **spatial extent** and choose the ID that best represents the **main body** of the described object across both views.

3. **Iterate if needed**:
   - If no candidate fully matches the query, return to the global view and reassess alternatives.
   - Repeat verification with camera images until you confidently identify the best match.

---

**Task**:
Select the object ID of the target class: **{target_class}**  
Query description: "{text}"

Object IDs to choose from: {object_id_list}

---

**Output format (JSON)**:
{{
  "reasoning": "Explain how you analyzed spatial relationships (view-dependent vs view-independent), cross-verified the object across views, and selected the best-matched ID.",
  "object_id": ID  // e.g., 10
}}
"""

all_views = ["top", "left", "right", "up", "down"] 