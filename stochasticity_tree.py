import numpy as np

from ale_python_interface import ALEInterface
import copy

# test = 'loadROM'
# test = 'restoreState'
test = 'restoreSystemState'

frame_skip = 4
bunch = 200
sequence = 800

def main():
    result = {
        'name':[],
        'grouped_num':[],
        'distribution':[],
    }
    result_str = ''

    # all_game_list = ['air_raid-n', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis']
    # all_game_list = ['bank_heist', 'battle_zone', 'beam_rider', 'berzerk-n', 'bowling', 'boxing', 'breakout', 'carnival-n']
    # all_game_list = ['centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk']
    # all_game_list = ['elevator_action-n', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar']
    # all_game_list = ['hero', 'ice_hockey', 'jamesbond', 'journey_escape-n', 'kangaroo', 'krull', 'kung_fu_master']
    # all_game_list = ['montezuma_revenge-n', 'ms_pacman', 'name_this_game', 'phoenix-n', 'pitfall-n', 'pong', 'pooyan-n']
    # all_game_list = ['private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing-n']
    # all_game_list = ['solaris-n', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down']
    # all_game_list = ['venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge-n', 'zaxxon']

    # all_game_list = ['pong', 'assault','ms_pacman']
    all_game_list = ['assault']

    for game in all_game_list:

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
        if test in ['restoreSystemState']:
            state_after_reset = env_father.cloneSystemState()

        print('=====================================================')
        try:
            action_sequence = np.load(
                'action_sequence_{}_{}.npy'.format(
                    sequence,
                    game,
                )
            )
            print('action_sequence loaded')
        except Exception as e:
            '''generate a sequence of actions'''
            action_sequence = np.random.randint(
                len(env_father.getMinimalActionSet()),
                size = sequence,
            )
            np.save(
                'action_sequence_{}_{}.npy'.format(
                    sequence,
                    game,
                ),
                action_sequence,
            )
            print('action_sequence generated')
        print('=====================================================')

        bunch_obs = []
        distribution = []
        samples = []
        for bunch_i in range(bunch):

            env_temp = env_father
            env_temp.setInt(b'random_seed', 3)
            if test in ['loadROM']:
                env_temp.loadROM(game_path)
                env_temp.reset_game()
            elif test in ['restoreState']:
                env_temp.restoreState(state_after_reset)
            elif test in ['restoreSystemState']:
                env_temp.restoreSystemState(state_after_reset)

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
        try:
            game_list += [game]
        except Exception as e:
            game_list = [game]
        try:
            grouped_num_list += [grouped_num]
        except Exception as e:
            grouped_num_list = [grouped_num]

    print(result_str)
    print('===============')
    for game_i in range(len(game_list)):
        print(game_list[game_i])
    for grouped_num_i in range(len(grouped_num_list)):
        print(grouped_num_list[grouped_num_i])

if __name__ == "__main__":
    main()
