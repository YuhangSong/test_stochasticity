import numpy as np

from ale_python_interface import ALEInterface

test = 'loadROM'
# test = 'restoreState'

frame_skip = 4
bunch = 200
sequence = 50

def main():
    result = {
        'name':[],
        'grouped_num':[],
        'distribution':[],
    }
    result_str = ''

    # game_list = ['air_raid-n', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis']
    # game_list = ['bank_heist', 'battle_zone', 'beam_rider', 'berzerk-n', 'bowling', 'boxing', 'breakout', 'carnival-n']
    # game_list = ['centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk']
    # game_list = ['elevator_action-n', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar']
    # game_list = ['hero', 'ice_hockey', 'jamesbond', 'journey_escape-n', 'kangaroo', 'krull', 'kung_fu_master']
    # game_list = ['montezuma_revenge-n', 'ms_pacman', 'name_this_game', 'phoenix-n', 'pitfall-n', 'pong', 'pooyan-n']
    # game_list = ['private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing-n']
    # game_list = ['solaris-n', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down']
    # game_list = ['venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge-n', 'zaxxon']

    game_list = ['pong', 'assault','ms_pacman']

    for game in game_list:

        if '-n' in game:
            '''games that are not in the nature DQN list'''
            continue

        import atari_py
        game_path = atari_py.get_game_path(game)
        game_path = str.encode(game_path)

        env_father = ALEInterface()
        env_father.setFloat('repeat_action_probability'.encode('utf-8'), 0.0)
        env_father.setInt(b'random_seed', 3)
        env_father.loadROM(game_path)
        env_father.reset_game()

        if test in ['restoreState']:
            state_after_reset = env_father.cloneState()

        '''generate a sequence of actions'''
        action_sequence = np.random.randint(
            len(env_father.getMinimalActionSet()),
            size = sequence,
        )

        bunch_obs = []
        distribution = []
        samples = []
        for bunch_i in range(bunch):

            env_temp = ALEInterface()
            env_temp.setFloat('repeat_action_probability'.encode('utf-8'), 0.0)
            env_temp.setInt(b'random_seed', bunch_i)
            if test in ['loadROM']:
                env_temp.loadROM(game_path)
                env_temp.reset_game()
            elif test in ['restoreState']:
                env_temp.loadROM(game_path) # restoreState without calling loadROM first will cause Segmentation fault (core dumped)
                env_temp.restoreState(state_after_reset)

            # just to make sure
            env_temp.setFloat('repeat_action_probability'.encode('utf-8'), 0.0)
            env_temp.setInt(b'random_seed', bunch_i)

            for sequence_i in range(sequence):
                for frame_skip_i in range(frame_skip):
                    env_temp.act(
                        env_father.getMinimalActionSet()[
                            action_sequence[sequence_i]
                        ]
                    )
                if env_temp.game_over():
                    env_temp.reset_game()

            obs = env_temp.getScreenRGB()

            samples += [obs]
            found_at_bunch = -1
            if_has_identical_one = False
            max_value = 0
            for bunch_obs_i in range(len(bunch_obs)):
                obs_in_bunch = bunch_obs[bunch_obs_i]
                max_value = np.max(
                    np.abs(
                        obs-obs_in_bunch
                    )
                )
                if max_value < 1:
                    found_at_bunch = bunch_obs_i
                    if_has_identical_one = True
                    distribution[found_at_bunch] += 1
                    break

            if if_has_identical_one is False:
                bunch_obs += [obs]
                distribution += [1]

        grouped_num = len(bunch_obs)
        result_str = '{}game:{} grouped_num:{} distribution:{} \n'.format(
            result_str,
            game,
            grouped_num,
            distribution,
        )

    print(result_str)

if __name__ == "__main__":
    main()
