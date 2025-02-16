import numpy as np
import nural_network as nn
import time
import matplotlib.pyplot as plt
def simulate_turn(round_score, rolls, dice_left, depth=5):
    """
    Recursively explores up to `depth` turns ahead to estimate the best possible move.
    """
    if depth == 0:
        return 0  # Stop recursion after reaching depth

    best_expected_score = 0
    best_move = None

    # Roll dice
    if dice_left == 0:
        rolls = np.random.randint(1, 7, size=6)
        dice_left = 6
    elif depth < 4 and dice_left < 6:
        
        rolls = np.random.randint(1, 7, size=dice_left)
        rolls = np.pad(rolls, (0, 6 - len(rolls)), constant_values=7)  # Fill empty spots with 7
        np.random.shuffle(rolls)
      

    # Find all valid moves
    possible_moves = []
    for i in range(1, 7):  # Dice values 1-6
        count = np.count_nonzero(rolls == i)
        if count > 0:
            chosen_rolls = [i] * count
            move_score = score(chosen_rolls)
            if move_score > 0:
                possible_moves.append((chosen_rolls, move_score))

    if not possible_moves:
        return [0, [], [], False]  # Bust

    # Explore each move
    for chosen_rolls, move_score in possible_moves:
        dice_remaining = max(0, 6 if len(chosen_rolls) == 6 else dice_left - len(chosen_rolls))
        new_rolls = np.random.randint(1, 7, size=dice_remaining) if dice_remaining > 0 else np.array([])
        new_rolls = np.pad(new_rolls, (0, 6 - len(new_rolls)), constant_values=7)

        future_score = simulate_turn(round_score + move_score, new_rolls, dice_remaining, depth - 1)
        future_score = future_score[0] if isinstance(future_score, list) else future_score
        expected_score = move_score + future_score

        if expected_score > best_expected_score:
            best_expected_score = expected_score
            best_move = chosen_rolls

    # Collect dice indices
    indices = []
    for die in best_move:
        for idx, roll in enumerate(rolls):
            if roll == die and idx not in indices:
                indices.append(idx)
                break

    indices = np.array(indices)
    reroll = len(possible_moves) > 2

    return [best_expected_score, indices, best_move, reroll]


def one(ones):
    if ones == 1:
        return 100
    elif ones == 2:
        return 200
    elif ones == 3:
        return 1000
    elif ones == 4:
        return 2000
    elif ones == 5:
        return 4000
    elif ones == 6:
        return 8000
    else:
        return ones*100
def two(twos):
    if twos == 3:
        return 200
    elif twos == 4:
        return 400
    elif twos == 5:
        return 800
    elif twos == 6:
        return 1600
    else:
        return 0
def three(threes):
    if threes == 3:
        return 300
    elif threes == 4:
        return 600
    elif threes == 5:
        return 1200
    elif threes == 6:
        return 2400
    else:
        return 0
def four(fours):
    if fours == 3:
        return 400
    elif fours == 4:
        return 800
    elif fours == 5:
        return 1600
    elif fours == 6:
        return 3200
    else:
        return 0
def five(fives):
    if fives == 1:
        return 50
    elif fives == 2:
        return 100
    elif fives == 3:
        return 500
    elif fives == 4:
        return 1000
    elif fives == 5:
        return 2000
    elif fives == 6:
        return 4000
    else:
        return fives*50
def six(sixes):
    if sixes == 3:
        return 600
    elif sixes == 4:
        return 1200
    elif sixes == 5:
        return 2400
    elif sixes == 6:
        return 4800
    else:
        return 0
def score(chosen_rolls,test=False):
    ones=0
    twos=0
    threes=0
    fours=0
    fives=0
    sixes=0
    sevens=0
    for num in chosen_rolls:
        if num == 1:
            ones=ones+1
        elif num == 2:
            twos=twos+1
        elif num == 3:
            threes=threes+1
        elif num == 4:
            fours=fours+1
        elif num == 5:
            fives=fives+1
        elif num == 6:
            sixes=sixes+1
        elif num == 7:
            sevens=sevens+1
    score = [one(ones) , two(twos) , three(threes) , four(fours) , five(fives) , six(sixes),0]
    arr=[ones,twos,threes,fours,fives,sixes,sevens]
    for i in range(6):
        if arr[i] ==0:
            continue
        else:
            if score[i] == 0 and test == False:
                return 0
    score=np.sum(np.array(score))
    
    return score
def train():
    diff_scores_avg=[]
    val=np.arange(0, 100000)
    num_die=6
    for i in val:
        if num_die==0:
            num_die=6
        notsix=False

        rolls = np.random.randint(1, 7, size=num_die)
        if len(rolls) < 6:
            notsix=True
        while notsix==True:
            rolls=np.append(rolls,7)
            if len(rolls) < 6:
                notsix=True
            else:
                notsix=False
        np.random.shuffle(rolls)
        #print(len(rolls),num_die)
        target_score=np.random.randint(15,100)*100
        player_score = np.random.randint(0, target_score/100)*100
        oponant_score = np.random.randint(0, target_score/100)*100
        round_score=np.random.randint(0, 1000/100)*100
        holdm = simulate_turn(round_score,rolls,num_die, depth=5)
        #print(holdm,round_score,rolls,num_die)
        future_score = holdm[0]
        best_move = holdm[1]
        reroll=holdm[3]
        if future_score==0:
            diff_scores_avg.append(0)
            continue
        #rolls=np.append(rolls,round_score)
        #rolls=np.append(rolls,player_score)
        #rolls=np.append(rolls,oponant_score)
        #rolls=np.append(rolls,target_score)
        #print(rolls)
        out=network.main(rolls)
        #print(out,best_move)
        bm=np.zeros(7)
        bm[best_move]=1
        if reroll==True:
            bm[6]=1
        num_die = num_die-len(best_move)
        #print(bm)
        if future_score==0:
            bm[6]=1
        network.ajusts(bm,rolls)
        diff_scores_avg.append(np.mean(bm-out))

        #print(np.mean(bm-out))
    plt.plot(val,diff_scores_avg)
    plt.xscale('log')
    plt.show()
def turn(player_score=0,opponent_score=0,target=0,ai=False):
    round_score = 0
    dice_left = 6
    while True:
        corect_input = False
        rolls = np.random.randint(1, 7, size=dice_left)
        print("You rolled: ", rolls)
        hold= simulate_turn(round_score,rolls,dice_left, depth=5)
        bets_future_score = hold[0]
        bets_indices = hold[1]
        print(rolls)
        chosen_rolls = []

        if score(rolls,test=True) == 0:
            print('You Bust!')
            round_score = 0
            break
        while corect_input == False:
            if ai==True:
                rolls = np.pad(rolls, (0, 6 - len(rolls)), constant_values=7)
                chosen=network.main(rolls)
 
                print(chosen,rolls)
                time.sleep(1)
                chosen_rolls = []
                for i in range(len(chosen)):
                    if chosen[i]>0.01 and i<7:
                        chosen_rolls.append(i)
                    elif i==7 and chosen[i]==1:
                        reroll=True
                    elif i==7 and chosen[i]==0:
                        reroll=False
            if ai==False:
                user_input = input("Enter the dice you want to keep separated by commas (e.g. 1, 2, 3): ")
        
                chosen_rolls = [int(num)-1 for num in user_input.split(',')]
            if np.max(chosen_rolls) > dice_left-1:
                print("please make a valid move!")
                continue
            dice_chosen = rolls[chosen_rolls]
            inputscore = score(dice_chosen)
            if inputscore == 0:
                print("please make a valid move!")
                continue
            else:
                corect_input = True
        dice_left = dice_left - len(chosen_rolls)
        if dice_left == 0:
            dice_left = 6
        print("You chose: ", dice_chosen)
        round_score += score(dice_chosen)
        print("Your score for this round so far  is: ", round_score, " and you have ", dice_left, " dice left" , " and you have ", player_score + round_score, " total points")
        if ai==True:
            if reroll==False:
                break
        if ai==False:
            user_input = input("Do you want to roll again? (y/n): ")
            if user_input == 'n':
                break
    print("Your score for this turn is: ", round_score, " and you have ", player_score + round_score, " total points")
    print()
    return round_score

def test(game_finish_score=10000):
    player_1_score = 0
    player_2_score = 0
    while player_1_score < game_finish_score and player_2_score < game_finish_score:
        print("Player 1's turn")
        player_1_score += turn(player_1_score,player_2_score,game_finish_score,ai=True)
        print("Player 1's total score is: ", player_1_score)
        print()
        print("Player 2's turn")
        player_2_score += turn(player_2_score,player_1_score,game_finish_score,ai=True)
        print("Player 2's total score is: ", player_2_score)
        print()

def game_loop(game_finish_score=10000):
    player_1_score = 0
    player_2_score = 0
    while player_1_score < game_finish_score and player_2_score < game_finish_score:
        print("Player 1's turn")
        player_1_score += turn(player_1_score,player_2_score,game_finish_score)
        print("Player 1's total score is: ", player_1_score)
        print()
        print("Player 2's turn")
        player_2_score += turn(player_2_score,player_1_score,game_finish_score)
        print("Player 2's total score is: ", player_2_score)
        print()
if __name__ == "__main__":

    network=nn.network([1,2,3,4,5,6],'dice_game_model')
    #network.loading()
    train()
    #test()
    network.finished()
    test()
    