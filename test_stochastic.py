import numpy as np

from ale_python_interface import ALEInterface
import copy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# test = 'loadROM'
test = 'setRAM'
# test = 'restoreState'
# test = 'restoreSystemState'

frame_skip = 4000
bunch = 100
sequence = 50

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

    # all_game_list = ['pong', 'assault','ms_pacman']
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

        if test in ['restoreState']:
            state_after_reset = env.cloneState()
        if test in ['restoreSystemState']:
            state_after_reset = env.cloneSystemState()
        if test in ['setRAM']:
            ram_after_reset = env.getRAM()
            state_after_reset = env.cloneSystemState()
            ram_candidate = np.load(
                './stochasticity_ram_mask/{}.npy'.format(
                    game
                ),
            )

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

        bunch_obs = []
        distribution = []
        episode_length = -1
        state_metrix = []
        ram_metrix = []
        for bunch_i in range(bunch):

            if test in ['loadROM']:
                env.setInt(b'random_seed', bunch_i)
                env.loadROM(game_path)
                env.reset_game()
            elif test in ['restoreState']:
                env.restoreState(state_after_reset)
            elif test in ['restoreSystemState']:
                env.restoreSystemState(state_after_reset)
            elif test in ['setRAM']:
                env.reset_game()
                env.restoreSystemState(state_after_reset)
                env.setRAM(ram_after_reset)
                env.setRAM(
                    env.getRAM()*(1-ram_candidate)+ram_candidate*(bunch_i%255)
                )

            state_sequence = []
            ram_sequence = []

            has_terminated = False
            for sequence_i in range(sequence):

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

                try:
                    print('[{}|{}|{}]'.format(bunch_i,sequence_i,episode_length))
                except Exception as e:
                    pass

                state_sequence += [env.getScreenRGB()]
                ram_sequence += [process_ram(env.getRAM())]

                if has_terminated:
                    break

            if sequence>0:
                if episode_length<0:
                    # raise Exception('Did not terminated')
                    print('# WARNING: Did not terminated')

            obs = env.getScreenRGB()

            state_metrix += [copy.deepcopy(state_sequence)]
            ram_metrix += [copy.deepcopy(ram_sequence)]

            if_has_identical_one = False
            for bunch_obs_i in range(len(bunch_obs)):
                max_value = np.max(
                    np.abs(
                        obs-bunch_obs[bunch_obs_i]
                    )
                )
                if max_value < 1:
                    if_has_identical_one = True
                    distribution[bunch_obs_i] += 1
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

        max_lenth = 0
        for bunch_i in range(len(state_metrix)):
            if len(state_metrix[bunch_i])>max_lenth:
                max_lenth = len(state_metrix[bunch_i])
        for bunch_i in range(len(state_metrix)):
            state_metrix[bunch_i] += ([np.zeros(shape=state_metrix[0][0].shape, dtype=state_metrix[0][0].dtype)]*(max_lenth-len(state_metrix[bunch_i])))
            ram_metrix  [bunch_i] += ([np.zeros(shape=ram_metrix  [0][0].shape, dtype=ram_metrix  [0][0].dtype)]*(max_lenth-len(state_metrix[bunch_i])))

        state_list = []
        state_metrix_id = np.zeros((len(state_metrix), len(state_metrix[0])), dtype=int)
        for bunch_i in range(len(state_metrix)):
            for sequence_i in range(len(state_metrix[0])):
                found_in_state_list = False
                for state_list_id in range(len(state_list)):
                    if np.max(state_list[state_list_id]-state_metrix[bunch_i][sequence_i])<1:
                        state_metrix_id[bunch_i][sequence_i] = state_list_id
                        found_in_state_list = True
                        break
                if not found_in_state_list:
                    state_list += [np.copy(state_metrix[bunch_i][sequence_i])]
                    state_metrix_id[bunch_i][sequence_i] = (len(state_list)-1)

        state_metrix_id_unsorted = np.copy(state_metrix_id)
        state_metrix_id = state_metrix_id.tolist()
        state_metrix_id.sort(key=lambda row: row[:], reverse=True)
        state_metrix_id = np.array(state_metrix_id)

        fig, ax = plt.subplots()
        im = ax.imshow(state_metrix_id)
        plt.show()
        plt.savefig(
            './results/{}_state_metrix_id.jpg'.format(game),
            dpi=600,
        )

        state_metrix_figure = np.zeros(((10+state_metrix[0][0].shape[0])*len(state_metrix),state_metrix[0][0].shape[1]*len(state_metrix[0]), state_metrix[0][0].shape[2]), dtype=state_metrix[0][0].dtype)
        ram_metrix_figure   = np.zeros(((5 +ram_metrix  [0][0].shape[0])*len(state_metrix),ram_metrix  [0][0].shape[1]*len(state_metrix[0]), ram_metrix  [0][0].shape[2]), dtype=ram_metrix  [0][0].dtype)

        ram_candidate = list(range(env.getRAMSize()))

        for bunch_i in range(len(state_metrix)):
            ram_metrix_figure  [((bunch_i)*(5 +ram_metrix  [0][0].shape[0])):(5 +(bunch_i)*(5 +ram_metrix  [0][0].shape[0])),:, 2] = 255
        for bunch_i in range(len(state_metrix)):
            for sequence_i in range(len(state_metrix[0])):
                state_metrix_figure[(10+(bunch_i)*(10+state_metrix[0][0].shape[0])):(bunch_i+1)*(10+state_metrix[0][0].shape[0]),(sequence_i)*state_metrix[0][0].shape[1]:(sequence_i+1)*state_metrix[0][0].shape[1]]=state_list[state_metrix_id[bunch_i][sequence_i]]
                for bunch_ii in range(state_metrix_id.shape[0]):
                    if np.max(state_metrix_id_unsorted[bunch_ii]-state_metrix_id[bunch_i])<1:
                        at_unsorted_bunch = bunch_ii
                        break
                ram_metrix_figure  [(5 +(bunch_i)*(5 +ram_metrix  [0][0].shape[0])):(bunch_i+1)*(5 +ram_metrix  [0][0].shape[0]),(sequence_i)*ram_metrix  [0][0].shape[1]:(sequence_i+1)*ram_metrix  [0][0].shape[1]]=ram_metrix[at_unsorted_bunch][sequence_i]

        for bunch_i in range(len(state_metrix)):
            for sequence_i in range(len(state_metrix[0])):
                if bunch_i > 0:
                    if state_metrix_id[bunch_i][sequence_i] != state_metrix_id[bunch_i-1][sequence_i]:
                        # draw a line to seperate the bunches
                        previous = ram_metrix_figure [(5 +(bunch_i-1)*(5 +ram_metrix  [0][0].shape[0])):((bunch_i)*(5 +ram_metrix  [0][0].shape[0])),sequence_i,0]
                        later = ram_metrix_figure [(5 +(bunch_i)*(5 +ram_metrix  [0][0].shape[0])):((bunch_i+1)*(5 +ram_metrix  [0][0].shape[0])),sequence_i,0]
                        delta = np.abs(previous-later)
                        state_metrix_figure[((bunch_i)*(10+state_metrix[0][0].shape[0])):(10+(bunch_i)*(10+state_metrix[0][0].shape[0])),(sequence_i)*state_metrix[0][0].shape[1]:, 0] = 255
                        ram_metrix_figure  [((bunch_i)*(5 +ram_metrix  [0][0].shape[0])):(5 +(bunch_i)*(5 +ram_metrix  [0][0].shape[0])),(sequence_i)*ram_metrix  [0][0].shape[1]:, 0] = 255
                        ram_metrix_figure  [((bunch_i)*(5 +ram_metrix  [0][0].shape[0])):(5 +(bunch_i)*(5 +ram_metrix  [0][0].shape[0])),(sequence_i)*ram_metrix  [0][0].shape[1]:, 1:] = 0


        from PIL import Image
        Image.fromarray(state_metrix_figure).save("./results/{}_state_metrix_figure.jpeg".format(
            game
        ))
        Image.fromarray(ram_metrix_figure.astype(state_metrix_figure.dtype)).save("./results/{}_ram_metrix_figure.jpeg".format(
            game
        ))

    print(result_str)
    print('===============')
    for game_i in range(len(game_list)):
        print(game_list[game_i])
    for grouped_num_i in range(len(grouped_num_list)):
        print(grouped_num_list[grouped_num_i])

if __name__ == "__main__":
    main()
