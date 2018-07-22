import random
import sys, math
import numpy as np
import time

from Box2D import *

from collections import deque



import gym
from gym import spaces
from gym.utils import seeding

from gym.envs.classic_control import rendering
#import rendering


from pyglet.window import key

from gym.utils import reraise

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")


STATE_W = 50
STATE_H = 50
STATE_SCALE = 10

FPS = 60

VIEWPORT_W = 300
VIEWPORT_H = 300
SCALE  = 30.0

WORLD_W = 15.0
WORLD_H = 15.0

AGENT_RADIUS = 0.5
FOOD_RADIUS = 0.3
WALL_THICKNESS = 0.5

CENTER_X = 0
CENTER_Y = 0

DEFAULT_SPEED = 2.0
DEFAULT_ANGLE = 0.25
AGENT_COLOR = (1,0,0)
AGENT_DOT_COLOR = (0,1,0)

FOOD_NUM = 20

EPISODE_STEPS = 2000

STATE_MODE = "gs_array"

def rad2vec(r):
    return b2Vec2(math.cos(r), math.sin(r))


class Cleaner(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'gs_array'],
    }

    # Set these in ALL subclasses
    action_space = spaces.Discrete(3)
    observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 1))


    class DrawCallback(b2QueryCallback):
        def __init__(self, draw_food, draw_wall, draw_charger, viewer):
            b2QueryCallback.__init__(self)
            self.draw_food = draw_food
            self.draw_wall = draw_wall
            self.draw_charger = draw_charger
            self.viewer = viewer
            self.render_queue = []

        def ReportFixture(self, fixture):
            # body = fixture.body
            self.render_queue.append(fixture)

            return True

        def Render(self):

            self.render_queue.sort(key=lambda x: x.body.userData)
            for fixture in self.render_queue:
                body = fixture.body

                if False:
                    pass
                elif body.userData == "charger":
                    self.draw_charger(body, self.viewer)
                elif body.userData == "food":
                    self.draw_food(body, self.viewer)
                elif body.userData == "wall":
                    self.draw_wall(body, self.viewer)
            return True

    class ContactListener(b2ContactListener):
        def __init__(self, owner):
            self.owner = owner
            b2ContactListener.__init__(self)
            self.destroy_list = []
            self.move_list = []
            self.wall_contact = False
            self.charger_contact = False

        def BeginContact(self, contact):
            A = contact.fixtureA
            B = contact.fixtureB

            if B.body.userData == "food":
                # self.destroy_list.append(B.body)
                rand_point = self.owner.get_free_point()
                self.move_list.append([B.body, rand_point])

            if B.body.userData == "wall" or A.body.userData == "wall":
                self.wall_contact = True

            if B.body.userData == "charger" or A.body.userData == "charger":
                self.charger_contact = True

        def EndContact(self, contact):
            A = contact.fixtureA
            B = contact.fixtureB

            if B.body.userData == "wall" or A.body.userData == "wall":
                self.wall_contact = False

            if B.body.userData == "charger" or A.body.userData == "charger":
                self.charger_contact = False

        def PreSolve(self, contact, oldManifold):
            pass

        def PostSolve(self, contact, impulse):
            pass

        def MoveBodies(self, world):
            n = len(self.move_list)

            for body, position in self.move_list:
                body.position = position

            self.move_list = []

            return n

        def DestroyBodies(self, world):
            n = len(self.destroy_list)

            for body in self.destroy_list:
                world.DestroyBody(body)

            self.destroy_list = []

            return n



    def __init__(self, summed_objectives=False, randomized=False, deterministic=False):
        # self.observation_space = spaces.Box(-high, high)

        self.deterministic = deterministic
        if self.deterministic:
            self._map_seed = 1
            self._seed(seed=self._map_seed)
        else:
            self._seed()
        self.randomized = randomized

        self.summed_objectives = summed_objectives

        self.contact_listener = self.ContactListener(self)
        self.world = Box2D.b2World((0, 0), doSleep=True, contactListener=self.contact_listener)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None

        self.state_rendered = False

        self.reward = 0.0
        self.prev_reward = 0.0

        self.static_objects = []
        self.dynamic_objects = []

        self.viewers = {"map": None, "state": None}
        self.extra_indicators = deque(maxlen=1)

        self.walls = []
        self.agent = None
        self.food = []
        self.chargers = []

        self.sim_stats = {}
        self.sim_stats['charge'] = 1.0

        self.state = None

        self._create_walls()

        if self.randomized:
            self.randomize_layout()
        else:
            self._add_obstacle(3, 3, 1.5, 1.5)
            self._add_obstacle(-3, -3, 1.5, 1.5)
            self._add_charger(-5, 5, 1, 1)
            self._add_charger(5, -5, 1, 1)

        self._create_agent()

        self.steps_to_reset = EPISODE_STEPS

        self.keys = key.KeyStateHandler()
        self.render_map = False

        self.last_reset = time.time()

        for _ in range(FOOD_NUM):
            self.create_random_food()

    def get_objectives(self):
        if self.summed_objectives:
            return ['main']
        else:
            return ['collision', 'clean', 'charge']

    def get_action_meanings(self):
        return ['left', 'straight', 'right']

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _reset(self):
        print("RESETING GAME")
        print("performance: %f steps per second", EPISODE_STEPS / (time.time() - self.last_reset))

        if self.deterministic:
            self._map_seed += 1
            self._seed(self._map_seed)

        if self.randomized:
            self.randomize_layout()

        self.last_reset = time.time()
        self.steps_to_reset = EPISODE_STEPS
        self.agent.position = self.get_free_point()
        self.agent.angle = 0

        self.sim_stats['charge'] = 1.0

        for f in self.food:
            f.position = self.get_free_point()

        return self._render(mode=STATE_MODE)


    def _create_walls(self):
        # four sides of the wall
        left_wall = self.world.CreateStaticBody(
            position=(CENTER_X - WORLD_W/2 - WALL_THICKNESS, CENTER_Y),
            shapes=b2PolygonShape(box=(WALL_THICKNESS, WORLD_H/2)),
            awake=False,
            userData = "wall",
        )

        top_wall = self.world.CreateStaticBody(
            position=(CENTER_X, CENTER_Y + WORLD_H/2 + WALL_THICKNESS),
            shapes=b2PolygonShape(box=(WORLD_W/2 + 2*WALL_THICKNESS, WALL_THICKNESS)),
            awake=False,
            userData = "wall",
        )

        bottom_wall = self.world.CreateStaticBody(
            position=(CENTER_X, CENTER_Y - WORLD_H/2 - WALL_THICKNESS),
            shapes=b2PolygonShape(box=(WORLD_W / 2 + 2 * WALL_THICKNESS, WALL_THICKNESS)),
            awake=False,
            userData = "wall",
        )

        right_wall = self.world.CreateStaticBody(
            position=(CENTER_X + WORLD_W/2 + WALL_THICKNESS, CENTER_Y),
            shapes=b2PolygonShape(box=(WALL_THICKNESS, WORLD_H/2)),
            awake=False,
            userData="wall",
        )

        self.walls += [left_wall, top_wall, bottom_wall, right_wall]

    def randomize_layout(self):
        for w in self.walls + self.chargers:
            self.world.DestroyBody(w)



        self.walls = []
        self.chargers = []

        self._create_walls()

        num_charges = random.randint(1, 2)
        for i in range(num_charges):
            self._add_charger(random.randint(-6, 6), random.randint(-6, 6), random.randint(1, 2), random.randint(1, 2))

        num_obstacles = random.randint(1, 5)
        for i in range(num_obstacles):
            self._add_obstacle(random.randint(-6, 6), random.randint(-6, 6), random.randint(1, 4), random.randint(1, 4))

    def _add_obstacle(self, x, y, a, b):
        obstacle = self.world.CreateStaticBody(
            position=(x, y),
            shapes=b2PolygonShape(box=(a, b)),
            awake=False,
            userData="wall",
        )

        self.walls += [obstacle]

    def _add_charger(self, x, y, a, b):
        charger = self.world.CreateStaticBody(
            position=(x, y),
            shapes=b2PolygonShape(box=(a, b)),
            awake=False,
            userData="charger",
        )

        charger.fixtures[0].sensor = True

        self.chargers += [charger]

    def _draw_wall(self, wall, viewer):
        for w in wall.fixtures:
            trans = w.body.transform
            path = [trans * v for v in w.shape.vertices]
            viewer.draw_polygon(path)

    def _draw_charger(self, wall, viewer):
        for w in wall.fixtures:
            trans = w.body.transform
            path = [trans * v for v in w.shape.vertices]
            viewer.draw_polygon(path, color=(0.5, 0.5, 0.5))

    def _create_agent(self):
        pos_x, pos_y = self.get_free_point()
        circle1 = b2CircleShape(pos=(0, 0), radius=AGENT_RADIUS)
        circle2 = b2CircleShape(pos=(AGENT_RADIUS/2, 0), radius=AGENT_RADIUS/6)
        self.agent = self.world.CreateDynamicBody(shapes=[circle1, circle2], position=(pos_x, pos_y), userData="agent")
        self.agent.angle = 0.0

    def _draw_agent(self, viewer):
        a_main = self.agent.fixtures[0]
        a_dot = self.agent.fixtures[1]

        color = (1, 1. - self.sim_stats['charge'], 1. - self.sim_stats['charge'])

        trans = a_main.body.transform
        t = rendering.Transform(translation=trans * a_main.shape.pos)
        viewer.draw_circle(a_main.shape.radius, 20, filled=True, color=color).add_attr(t)

        trans = a_dot.body.transform
        t = rendering.Transform(translation=trans * a_dot.shape.pos)
        viewer.draw_circle(a_dot.shape.radius, 20, filled=True, color=AGENT_DOT_COLOR).add_attr(t)

    def _create_food(self, x, y):
        circle1 = b2CircleShape(pos=(0, 0), radius=FOOD_RADIUS)
        food = self.world.CreateStaticBody(shapes=[circle1], position=(x, y), userData="food")
        food.fixtures[0].sensor = True
        return food

    def get_free_point(self):
        while (True):
            collide = False
            rand_x, rand_y = (WORLD_W - 1) * (random.random() - 0.5), (WORLD_H - 1) * (random.random() - 0.5)
            for f in self.walls:
                if f.fixtures[0].TestPoint((rand_x, rand_y)):
                    collide = True
                    break
            if not collide:
                break

        return (rand_x, rand_y)

    def create_random_food(self):
        f = self._create_food(*self.get_free_point())
        self.food.append(f)

    def _draw_food(self, f, viewer):
        a = f.fixtures[0]

        trans = a.body.transform
        t = rendering.Transform(translation=trans * a.shape.pos)

        viewer.draw_circle(a.shape.radius, 20, filled=True, color=(0, 0, 0)).add_attr(t)
        viewer.draw_circle(a.shape.radius/1.5, 20, filled=True, color=(1, 1, 1)).add_attr(t)
        viewer.draw_circle(a.shape.radius/2.5, 20, filled=True, color=(0, 0, 0)).add_attr(t)

    def _agent_move(self, speed, angle):
        self.agent.angle += angle
        self.agent.linearVelocity = rad2vec(self.agent.angle) * speed

    def _step(self, action):
        if self.steps_to_reset > 0 and self.sim_stats['charge'] > 0:
            if action == 0:
                self._agent_move(DEFAULT_SPEED, -DEFAULT_ANGLE)
            elif action == 1:
                self._agent_move(DEFAULT_SPEED, 0)
            elif action == 2:
                self._agent_move(DEFAULT_SPEED, DEFAULT_ANGLE)
            else:
                pass

            self._simulate()

            self.state = self._render(mode=STATE_MODE)

            # print(self.sim_stats["charge"])

            food_reward = float(self.sim_stats['destroyed_food_num'])
            wall_reward = -1. if self.contact_listener.wall_contact else 0.
            charge_reward = -1. if self.sim_stats["charge"] <= 0.1 else self.sim_stats['charged']

            if self.summed_objectives:
                # _r = [wall_reward, food_reward, charge_reward]
                # print(_r)
                reward = wall_reward + food_reward + charge_reward
            else:
                reward = dict()
                reward['collision'] = wall_reward
                reward['clean'] = food_reward
                reward['charge'] = charge_reward

            done = False

            # self._render_map()

            self.steps_to_reset -= 1
        else:
            if self.summed_objectives:
                reward = 0
            else:
                reward = {'collision': 0., 'clean': 0., 'charge': 0.}
            done = True

        if self.render_map:
            self._render()

        #check keys
        if self.keys[key.M]:
            print("M_on")
            self.keys[key.M] = False
            self.render_map = True
        elif self.keys[key.Q]:
            print("Q_on")
            self.keys[key.Q] = False
            self._render(close=True)
            self.render_map = False

        return self.state, reward, done, {}

    def set_extra_indicators(self, indicators):
        self.extra_indicators.append(indicators)

    def _render(self, mode='human', close=False):
        if close:
            for n, viewer in self.viewers.items():
                if viewer is not None:
                    viewer.close()
                    self.viewers[n] = None
            return

        if mode == "human":
            if self.viewers["map"] is None:
                self.viewers["map"] = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
                self.viewers["map"].window.set_vsync(False)
                self.viewers["map"].set_bounds(-VIEWPORT_W / SCALE, VIEWPORT_W / SCALE, -VIEWPORT_H / SCALE, VIEWPORT_H / SCALE)

                window = self.viewers["map"].window
                window.push_handlers(self.keys)

            v = self.viewers["map"]

            aabb = b2AABB(lowerBound=(-VIEWPORT_W / SCALE, -VIEWPORT_H / SCALE),
                          upperBound=(VIEWPORT_W / SCALE, VIEWPORT_H / SCALE))

        if mode == "rgb_array" or mode == "gs_array":
            if self.viewers["state"] is None:
                self.viewers["state"] = rendering.Viewer(STATE_W, STATE_H)
                self.viewers["state"].window.set_vsync(False)

                window = self.viewers["state"].window
                window.push_handlers(self.keys)

            v = self.viewers["state"]

            scroll_x = self.agent.position[0]
            scroll_y = self.agent.position[1]

            scale = STATE_SCALE
            angle = -self.agent.angle + b2_pi / 2

            tx = STATE_W / 2 - (scroll_x * scale * math.cos(angle) - scroll_y * scale * math.sin(angle))
            ty = 0 - (scroll_x * scale * math.sin(angle) + scroll_y * scale * math.cos(angle))

            trans = rendering.Transform(translation=(tx, ty), rotation=angle, scale=(scale, scale))

            v.transform = trans

            agent_pos = self.agent.position

            a = agent_pos + (2 * STATE_H / SCALE * math.sqrt(2) * math.sin(angle - 0.46364),
                             2 * STATE_H / SCALE * math.sqrt(2) * math.cos(angle - 0.46364))
            b = agent_pos + (2 * STATE_H / SCALE * math.sqrt(2) * math.sin(angle + 0.46364),
                             2 * STATE_H / SCALE * math.sqrt(2) * math.cos(angle + 0.46364))
            c = agent_pos + (STATE_W / SCALE * math.sin(angle - b2_pi / 2),
                             STATE_W / SCALE * math.cos(angle - b2_pi / 2))
            d = agent_pos + (STATE_W / SCALE * math.sin(angle + b2_pi / 2),
                             STATE_W / SCALE * math.cos(angle + b2_pi / 2))

            xs, ys = zip(*[a, b, c, d])
            minx, miny = min(xs), min(ys)
            maxx, maxy = max(xs), max(ys)

            aabb = b2AABB(
                lowerBound=(minx, miny),
                upperBound=(maxx, maxy))

        query = self.DrawCallback(self._draw_food, self._draw_wall, self._draw_charger, v)
        self.world.QueryAABB(query, aabb)
        query.Render()

        self._draw_agent(v)

        if len(self.extra_indicators) > 0 and mode == "human":
            ind_num = len(self.extra_indicators)
            lx, rx, uy, dy = -10, 0, 10, 9.5

            averaged = np.mean(np.array(self.extra_indicators), axis=0)
            for i, ind in enumerate(averaged):
                color = [0., 0., 0.]
                color[i%3] = 1.0
                drx = lx + (rx-lx)*ind
                v.draw_polygon([(lx, uy-i*0.5), (drx, uy-i*0.5), (drx, dy-i*0.5), (lx, dy-i*0.5)], color=color)


        if mode == "rgb_array":
            rgb_state = v.render(return_rgb_array=True)
            self.state = rgb_state
        elif mode == "gs_array":
            rgb_state = v.render(return_rgb_array=True)
            self.state = np.expand_dims(np.dot(rgb_state[..., :3], [0.299, 0.587, 0.114]), axis=2)
        else:
            v.render()

        return self.state


    def _simulate(self):
        timeStep = 1 / 20
        vel_iters, pos_iters = 6, 2

        self.destroy_list = []

        self.world.Step(timeStep, vel_iters, pos_iters)
        self.world.ClearForces()

        self.sim_stats['destroyed_food_num'] = self.contact_listener.MoveBodies(self.world)

        if self.contact_listener.charger_contact:
            self.sim_stats['charged'] = (1. - self.sim_stats['charge']) * 0.1
            self.sim_stats['charge'] += self.sim_stats['charged']
            self.sim_stats['charge'] = min(self.sim_stats['charge'], 1.0)

        else:
            self.sim_stats['charged'] = 0.

        self.sim_stats['charge'] -= 1.0 / 1000.0

        #self.sim_stats['destroyed_food_num'] = self.contact_listener.DestroyBodies(self.world)

        # for _ in range(self.sim_stats['destroyed_food_num']):
        #     self._create_food((WORLD_W - 1) * (random.random() - 0.5), (WORLD_H - 1) * (random.random() - 0.5))



from gym.envs.registration import registry, register, make, spec

register(
    id='CleanerNoFrameskipMultiObjective-v1',
    entry_point='cleaner:Cleaner',
    kwargs={'summed_objectives' : False},
)

register(
    id='CleanerNoFrameskipMultiObjectiveRandomized-v1',
    entry_point='cleaner:Cleaner',
    kwargs={'summed_objectives' : False, 'randomized' : True},
)

register(
    id='CleanerNoFrameskipMultiObjectiveRandomizedDeterministic-v1',
    entry_point='cleaner:Cleaner',
    kwargs={'summed_objectives' : False, 'randomized' : True, 'deterministic' : True},
)

register(
    id='CleanerNoFrameskipSingleObjective-v1',
    entry_point='cleaner:Cleaner',
    kwargs={'summed_objectives' : True},
)

register(
    id='CleanerNoFrameskipSingleObjectiveRandomized-v1',
    entry_point='cleaner:Cleaner',
    kwargs={'summed_objectives' : True, 'randomized' : True},
)







