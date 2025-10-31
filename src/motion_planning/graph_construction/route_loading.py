import sys
import json
import os

import carla
import sys
import os


import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
graph_construction_path = current_dir
sys.path.append(graph_construction_path)
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    print("Could not import GlobalRoutePlanner..")
    sys.exit(1)

routes_config = {
        ('Town01', 'T-Intersection'): ((200.0, 200.0), (336.0, 276.0)),
        ('Town03', 'Roundabout'): ((5.0, 100.0), (100.0, 8.0)),
        ('Town05', 'Highway1'): ((-228.0, 18.5), (-85.0, -189.0)), #29 layers
        ('Town05', 'Highway2'): ((-228.0, 18.5), (15.0, -189.0)), # 40 layers
        ('Town05', 'Highway3'): ((-228.0, 18.5), (107.0, -189.0)), #72 layers
        # ('Town05', 'Highway_3'): ((-228.0, 18.5), (100.0, -189.0)), # 72 layers
        ('Town05', 'Multiple Intersections'): ((90.0, -3.5), (-160.0, -3.0))
    }
class HDMapLoader:
    def __init__(self, map_id):
        """
        Initialize the HDMapLoader.
        
        :param map_source: 'town' or 'opendrive'
                           - 'town': load a built-in world by town name.
                           - 'opendrive': load a map from an OpenDRIVE (.xodr) file.
        :param map_identifier: For 'town', the town name (e.g., 'Town01'). For 'opendrive', the file path.
        """
        self.map_id = map_id
        self.client = None
        self.world = None
        self.map = None
        self.route = None


    # function to connect client to the server
    def connect_to_server(self, host='localhost', port=2000):
        """
        Connect to the CARLA server.
        """
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)


    def load_map(self):
        """
        Load the map either from a built-in town or an OpenDRIVE file.
        """
        if self.client is None:
            raise RuntimeError("Client is not connected to the server.")
        
        self.world = self.client.load_world(self.map_id)
        
    def get_map(self):
        """
        Retrieve the CARLA map object from the loaded world.
        """
        if self.world is None:
            raise RuntimeError("World is not loaded.")
        
        self.map = self.world.get_map()
        return self.map

    def plan_route(self, start_location, goal_location, sampling_resolution=1):
        """
        Plan a route using CARLA's GlobalRoutePlanner.
        
        :param start_location: carla.Location for the start point.
        :param goal_location: carla.Location for the destination.
        :param sampling_resolution: Distance (in meters) between waypoints.
        :return: A list of tuples (waypoint, road_option) representing the route.
        """
        if self.map is None:
            raise RuntimeError("Map is not loaded.")
        
        grp = GlobalRoutePlanner(self.map, sampling_resolution)
        print(start_location)
        print("Goal is: "+str(goal_location))
        # The trace_route method returns a list of (waypoint, road_option) tuples.
        route = grp.trace_route(carla.Location(start_location),carla.Location(goal_location) )
        return route

    

    def get_registered_route(self, map_id):
        """
        Get the registered route for the specified map.
        """
       
        try:
            with open('routes.json', 'r') as f:
                routes = json.load(f)
        except FileNotFoundError:
            raise RuntimeError("No routes found. Please register a route first.")
        
        # Find the route for the specified map_id
        for route in routes:
            if route['map_id'] == map_id:
                return route['route_wps']   # Return the waypoints of the route if found                
        raise RuntimeError(f"No route found for map ID: {map_id}")  

        




    
    def save_route(self, route, map_id, route_description):

        routes_file = 'routes.json'

        # Step 1: Load existing routes safely
        if os.path.exists(routes_file) and os.path.getsize(routes_file) > 0:
            with open(routes_file, 'r') as f:
                try:
                    routes = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: routes.json is corrupted. Starting fresh.")
                    routes = []
        else:
            routes = []

        # Step 2: Convert waypoints to serializable form
        route_wps_serialized = [
            {
                'x': wp.transform.location.x,
                'y': wp.transform.location.y,
                'z': wp.transform.location.z,
                'yaw': wp.transform.rotation.yaw
            }
            for (wp, _) in route
        ]

        # Step 3: Construct the new route dictionary
        route_data = {
            'map_id': map_id,
            'route_start': [
                route[0][0].transform.location.x,
                route[0][0].transform.location.y
            ],
            'route_goal': [
                route[-1][0].transform.location.x,
                route[-1][0].transform.location.y
            ],
            'route_wps': route_wps_serialized,
            'route_description': route_description
        }

        # Step 4: Append and save
        routes.append(route_data)
        with open('routes.json', 'w') as f:
            json.dump(routes, f, indent=4)

    def vis_route_server(self, route):
        """
        Visualize the route on the server.
        """
        if self.world is None:
            raise RuntimeError("World is not loaded.")
        
        i = 0
        # print(route.keys())
        for (wp,_) in route:
            if i == 0:
                color = carla.Color(r=255, g=0, b=10)
                life_time = 120.0
                box = carla.BoundingBox(wp.transform.location, carla.Vector3D(0.5, 0.5, 0.5))
                self.world.debug.draw_box(box,
                          wp.transform.rotation,
                          thickness=1.0,
                          color=color,
                          life_time=life_time,
                          persistent_lines=False)
            elif i==len(route)-1:
                color = carla.Color(r=0, g=255, b=0)
                life_time = 120.0   
                box = carla.BoundingBox(wp.transform.location, carla.Vector3D(0.5, 0.5, 0.5))
                self.world.debug.draw_box(box,
                          wp.transform.rotation,
                          thickness=1.0,
                          color=color,
                          life_time=life_time,
                          persistent_lines=False)
            else:
                color = carla.Color(r=0, g=0, b=255)
                life_time = 120.0
                self.world.debug.draw_string(wp.transform.location, 'O',
                                    draw_shadow=False,
                                    color=color,
                                    life_time=life_time,
                                    persistent_lines=True)
            i += 1

        # Example usage# Get the spectator
        spectator = self.world.get_spectator()

        # Choose a location to center the bird's-eye view — e.g., midpoint of your route
        mid_index = len(route) // 2
        midpoint = route[mid_index][0].transform.location  # Or .location if it's a raw Location object

        # Set a bird’s eye view transform
        height = 180.0  # Altitude for the camera (adjust as needed)
        bird_eye_location = carla.Location(midpoint.x, midpoint.y, midpoint.z + height)
        bird_eye_rotation = carla.Rotation(pitch=0, yaw=270, roll=0)  # Looking straight down

        # Apply the new transform to the spectator
        spectator.set_transform(carla.Transform(bird_eye_location, bird_eye_rotation))



def load_and_plan_route(map_id, road_scenario, start_coords, goal_coords, sampling_resolution):
    map_loader = HDMapLoader(map_id)
    map_loader.connect_to_server()
    map_loader.load_map()
    
    spawn_points = map_loader.get_map().get_spawn_points()

    # Use the first spawn point just for z-coordinate and other defaults
    start_location = carla.Location(spawn_points[0].location)
    goal_location  = carla.Location(spawn_points[0].location)

    start_location.x, start_location.y = start_coords
    goal_location.x, goal_location.y = goal_coords
    if road_scenario in ['Highway1', 'Highway2','Highway3']:
        start_location.z = 10.0
        goal_location.z = 10.0
    

    route = map_loader.plan_route(start_location=start_location, goal_location=goal_location, sampling_resolution=sampling_resolution)
    map_loader.vis_route_server(route)

    # Optionally save
    # map_loader.save_route(route, map_id, road_scenario)

    return route, map_loader.world



if __name__ == "__main__":

    map_id = 'Town05'
    road_scenario = 'Multiple-intersections'
    

    if (map_id, road_scenario) in routes_config:
        start, goal = routes_config[(map_id, road_scenario)]
        route,world = load_and_plan_route(map_id, road_scenario, start, goal)
        