import numpy as np
import cv2
import copy


class DrawingEnv_Digit:
    def __init__(self, params: dict) -> None:
        self.size = params["size"]
        self.line_color = params["line_color"]
        self.line_width = params["line_width"]
        self.initial_position = params["initial_position"]
        self.max_step = params["max_step"]
        self.mask = None
        self.prev_position = self.initial_position
        self._itr = 0

    def get_image(self, mask):
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for ch in range(3):
            image[:, :, ch] = mask * self.line_color[ch]

        return image

    def init(self, initial_position=None):
        self.mask = np.zeros((self.size, self.size), dtype=np.uint8)
        image = self.get_image(self.mask)
        self._itr = 0
        if initial_position == None:
            self.prev_position = self.initial_position
        else:
            self.prev_position = initial_position
        observation = dict(image=image, mask=self.mask, position=self.prev_position)
        return observation

    def step(self, position):
        if (self.prev_position[2] >= 1) and (position[2] >= 1):
            _mask = cv2.line(
                img=self.mask,
                pt1=(int(self.prev_position[0]), int(self.prev_position[1])),
                pt2=(int(position[0]), int(position[1])),
                color=1,
                thickness=self.line_width,
                lineType=cv2.LINE_AA,
            )
        else:
            _mask = self.mask
        self.mask = copy.deepcopy(_mask)
        image = self.get_image(self.mask)
        self.prev_position = position
        observation = dict(
            image=image,
            mask=self.mask,
            position=position,
        )
        reward = 0
        done = False
        info = dict()
        self._itr += 1
        if self._itr >= self.max_step:
            done = True
        return observation, reward, done, info


class DrawingEnv_Body:
    def __init__(self, params: dict) -> None:
        self.size = params["size"]
        self.initial_position = params["initial_position"]
        self.max_step = params["max_step"]
        self._itr = 0
        self.im_arm = cv2.imread("figs4env/arm_ex.png")[:, :, [2, 1, 0]]
        self.pos_pen = np.array((62, 30))

    def get_image(self, pos, allow_outrange=True):
        img_body = np.zeros((self.size, self.size, 3), np.uint8)
        dx, dy = np.round(pos[:2] - self.pos_pen).astype(np.int16)
        if allow_outrange:
            dx_base = 0
            if dx < 0:
                dx_base = -dx
                dx = 0
            dy_base = 0
            if dy < 0:
                dy_base = -dy
                dy = 0
            y, x, ch = img_body[dy:, dx:].shape
            dy_end, dx_end = self.im_arm[dy_base : y + dy_base, dx_base : x + dx_base].shape[:2]
            img_body[dy : dy + dy_end, dx : dx + dx_end] = self.im_arm[dy_base : y + dy_base, dx_base : x + dx_base]
        else:
            y, x, ch = img_body[dy:, dx:].shape
            if dy < 0 or dx < 0:
                raise NotImplementedError(f"dy (={dy}) < 0 or dx (={dx}) < 0")
            img_body[dy:, dx:] = self.im_arm[:y, :x]

        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        mask[img_body.sum(2) > 0] = 1

        return img_body, mask

    def init(self, initial_position=None):
        self._itr = 0
        if initial_position == None:
            initial_position = self.initial_position
        img, mask = self.get_image(initial_position)
        observation = dict(
            image=img,
            mask=mask,
            position=initial_position,
        )
        return observation

    def step(self, position):
        img, mask = self.get_image(position)
        observation = dict(
            image=img,
            mask=mask,
            position=position,
        )
        reward = 0
        done = False
        info = dict()
        self._itr += 1
        if self._itr >= self.max_step:
            done = True
        return observation, reward, done, info


class DrawingEnv_Moc1:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.digit_area_origin = np.array((self.params["digit_area"][0], self.params["digit_area"][1]))
        self.digit_area_size = np.array(
            (
                self.params["digit_area"][2] - self.params["digit_area"][0],
                self.params["digit_area"][3] - self.params["digit_area"][1],
            )
        )

        self.size = params["size"]
        params["initial_position"] = self.trans_pos(params["initial_position"])
        self.env_digit = DrawingEnv_Digit(params)
        self.env_body = DrawingEnv_Body(params)
        self.obs_residual = dict(image=cv2.imread("figs4env/background.png")[5:-5, 5:-5, [2, 1, 0]])

    def get_mask(self, obs_digit, obs_body):
        mask = np.zeros((self.size, self.size, 3))
        mask[:, :, 2] = obs_body["mask"].astype(np.float32)
        mask[:, :, 1] = np.clip(obs_digit["mask"].astype(np.float32) - obs_body["mask"].astype(np.float32), 0, 1)
        mask[:, :, 0] = np.clip(
            np.ones((self.size, self.size))
            - obs_digit["mask"].astype(np.float32)
            - obs_body["mask"].astype(np.float32),
            0,
            1,
        )
        mask = np.clip(mask, 0, 1).astype(np.uint8)
        return mask

    def marge_image(self, obs_digit, obs_body, obs_residual, mask):
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        image += obs_residual["image"] * np.expand_dims(mask[:, :, 0], -1)
        image += obs_digit["image"] * np.expand_dims(mask[:, :, 1], -1)
        image += obs_body["image"] * np.expand_dims(mask[:, :, 2], -1)

        return image

    def merge_observation(self, obs_digit, obs_body):
        mask = self.get_mask(
            obs_digit=obs_digit,
            obs_body=obs_body,
        )
        image = self.marge_image(
            obs_digit=obs_digit,
            obs_body=obs_body,
            obs_residual=self.obs_residual,
            mask=mask,
        )
        observation = dict(
            image=image,
            image_digit=obs_digit["image"],
            image_body=obs_body["image"],
            image_residual=self.obs_residual["image"],
            mask=mask,
            position=obs_digit["position"],
        )
        return observation

    def trans_pos(self, position):
        position[:2] = self.digit_area_origin + position[:2] * self.digit_area_size
        return position

    def init(self, initial_position=None):
        obs_digit = self.env_digit.init()
        obs_body = self.env_body.init()

        observation = self.merge_observation(obs_digit=obs_digit, obs_body=obs_body)
        return observation

    def step(self, position):
        pos = self.trans_pos(position)
        obs_digit, _, _, _ = self.env_digit.step(pos)
        obs_body, _, _, _ = self.env_body.step(pos)
        observation = self.merge_observation(obs_digit=obs_digit, obs_body=obs_body)
        reward = 0
        done = False
        info = dict()
        return observation, reward, done, info
