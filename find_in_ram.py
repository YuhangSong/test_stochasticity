import numpy as np

from ale_python_interface import ALEInterface
import copy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

frame_skip = 4
bunch = 1000
sequence = 5000

spaces = '                                                     '

def clear_print_line():
    print(spaces,end="\r")

def clear_print(string_to_print):
    clear_print_line()
    print(string_to_print,end="\r")

def process_ram(ram):
    temp = np.transpose(np.stack([ram.tolist()]*int(ram.shape[0]/210.0*160.0)))
    temp = np.expand_dims(temp, axis=2)
    temp = np.concatenate((temp,temp,temp), axis=2)
    return temp

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

    all_game_list = ['assault']

    for game in all_game_list:

        if '-n' in game:
            '''games that are not in the nature DQN list'''
            continue

        import atari_py
        game_path = atari_py.get_game_path(game)
        game_path = str.encode(game_path)

        env = ALEInterface()
        env.setFloat('repeat_action_probability'.encode('utf-8'), 0.0)

        env.setInt(b'random_seed', 3)
        env.loadROM(game_path)
        env.reset_game()

        print('=====================================================')
        try:
            action_sequence = np.load(
                './action_sequence/action_sequence_{}_{}.npy'.format(
                    sequence,
                    game,
                )
            )
            print('action_sequence loaded')
        except Exception as e:
            '''generate a sequence of actions'''
            action_sequence = np.random.randint(
                len(env.getMinimalActionSet()),
                size = sequence,
            )
            np.save(
                './action_sequence/action_sequence_{}_{}.npy'.format(
                    sequence,
                    game,
                ),
                action_sequence,
            )
            print('action_sequence generated')
        print('=====================================================')

        state_sequence_base = []
        ram_sequence_base = []
        has_terminated = False
        for sequence_i in range(sequence):

            state_sequence_base += [env.getScreenRGB()]
            ram_sequence_base += [env.getRAM()]

            for frame_skip_i in range(frame_skip):
                if not has_terminated:
                    env.act(
                        env.getMinimalActionSet()[
                            action_sequence[sequence_i]
                        ]
                    )
                    if env.game_over():
                        episode_length = sequence_i
                        has_terminated = True
                if has_terminated:
                    break

            if has_terminated:
                break

        ram_candidate = np.ones((env.getRAMSize()),dtype=np.uint8)


        state_sequence_branch = []
        ram_sequence_branch = []
        for bunch_i in range(bunch):

            env.setInt(b'random_seed', bunch_i)
            env.loadROM(game_path)
            env.reset_game()

            has_terminated = False
            for sequence_i in range(sequence):

                state_sequence_branch += [env.getScreenRGB()]
                ram_sequence_branch += [env.getRAM()]


                if sequence_i>0:
                    max_value = np.max(
                        np.abs(
                            env.getScreenRGB()-state_sequence_base[sequence_i]
                        )
                    )
                    if max_value > 0:
                        delta_ram = np.sign(np.abs(ram_sequence_branch[sequence_i-1]-ram_sequence_base[sequence_i-1]))
                        ram_candidate *= delta_ram
                        remain = np.sum(ram_candidate)
                        print('remain {} bytes'.format(remain))
                        if remain <= 1:
                            if remain == 1:
                                print(ram_candidate)
                                np.save(
                                    './stochasticity_ram_mask/{}.npy'.format(
                                        game
                                    ),
                                    ram_candidate,
                                )
                                raise Exception('done')
                            else:
                                raise Exception('error')
                        has_terminated = True

                if has_terminated:
                    break

                for frame_skip_i in range(frame_skip):
                    if not has_terminated:
                        env.act(
                            env.getMinimalActionSet()[
                                action_sequence[sequence_i]
                            ]
                        )
                        if env.game_over():
                            has_terminated = True
                    if has_terminated:
                        break

                if has_terminated:
                    break

if __name__ == "__main__":
    main()
