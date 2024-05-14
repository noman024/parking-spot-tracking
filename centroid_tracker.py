import numpy as np
# from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize the tracker with the given parameters.

        Parameters:
            max_disappeared (int): The maximum number of consecutive frames an object is allowed to be marked as "disappeared" before being deregistered.
            max_distance (int): The maximum Euclidean distance an object centroid is allowed to move in order to match it with a new detection.

        Returns:
            None
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance  # Maximum distance for object association

    def register(self, centroid):
        """
        A method to register a centroid with a unique object ID.

        Parameters:
            centroid: The centroid to be registered.

        Returns:
            object_id: The unique object ID assigned to the centroid.
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id

    def update(self, centroids):
        """
        Update the centroid tracker with a list of centroids.

        Parameters:
            centroids (List[Tuple[float, float]]): A list of centroids to update the tracker with.

        Returns:
            Dict[int, Tuple[float, float]]: A dictionary mapping object IDs to their corresponding centroids.

        Description:
            This method updates the centroid tracker with the given list of centroids. It first checks if the list is empty,
            and if so, it marks all objects as disappeared and deregisters them if they have been marked as disappeared for more
            than the specified maximum number of consecutive frames. It then initializes a set to keep track of the centroid
            indices that have been used in updating existing objects. Next, it iterates over the existing objects and checks
            the distance between their centroid and each centroid in the input list. If the distance is less than the
            specified maximum distance, it updates the object's centroid with the corresponding centroid from the input list
            and adds the centroid index to the set of used centroids. After updating the existing objects, it registers
            new objects for any centroids that have not been used in updating the existing objects. It then checks for
            disappeared objects by iterating over the object IDs and removing any IDs that are not present in the updated
            objects dictionary. If an object has been marked as disappeared for more than the specified maximum number of
            consecutive frames, it deregisters the object. Finally, it returns the updated objects dictionary.
        """
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # Initialize an array to keep track of used centroid indices
        used_centroids = set()

        # Update existing objects if their centroid is close to a new centroid
        for object_id, centroid in self.objects.items():
            distances = [np.linalg.norm(np.array(centroid) - np.array(new_centroid)) for new_centroid in centroids]
            min_distance = min(distances)
            if min_distance < self.max_distance:
                index = distances.index(min_distance)
                self.objects[object_id] = centroids[index]
                used_centroids.add(index)

        # Register new objects for centroids not used in updating existing objects
        unused_centroids = set(range(len(centroids))) - used_centroids
        for centroid_index in unused_centroids:
            object_id = self.register(centroids[centroid_index])

        # Check for disappeared objects
        for object_id in list(self.disappeared.keys()):
            if object_id not in self.objects:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        return self.objects
