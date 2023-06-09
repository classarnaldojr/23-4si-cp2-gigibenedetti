import cv2
import mediapipe as mp

video = cv2.VideoCapture('pedra-papel-tesoura.mp4')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_distance(point1, point2):
    #distância = raiz quadrada de ((x2 - x1)² + (y2 - y1)²)
    #A distância euclidiana é uma medida usada para calcular a distância entre dois pontos em um espaço. 
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

def get_hand_gestures(hand_landmarks):
    landmarks = hand_landmarks.landmark

    dist1 = get_distance(landmarks[8], landmarks[12])#distância do dedo indicador do dedo do meio
    dist2 = get_distance(landmarks[8], landmarks[4])#distância do dedo indicador do dedão

    # calculo da distância entre os dedos dependendo da distancia dos pontos definidos o programa reconhece como pedra tesoura ou papel
    if dist1 < 0.04 and dist2 < 0.04:
        return "pedra"
    elif dist1 > 0.06 and dist2 > 0.06:
        return "tesoura"
    else:
        return "papel"


# identifica a tupla dos landmarks que representa as maos de dois jogadores
def get_players_hand(mhl):
    p1_hand, p2_hand = mhl
#essa função calcula qual jogador tem a mão mais próxima da esquerda, compara a coord X do ponto mais a esquerda da mao de cada jogador
# por isso utilizamos map para aplica a função lambda l: l.x a cada ponto da lista de landmarks de cada mão obtemos uma lista
#com todas as coord x's dos landmarks de cada mao depois a comparação é feita e atribuimos p1 e p2 a mão correspondente
#lambda l: l.x é uma forma de criar uma função anonima de forma rápida l é o objeto que ela recebe e x é o atributo que ela retorna
    (p1_hand, p2_hand) = (p1_hand, p2_hand) if min(list(
        map(lambda l: l.x, p1_hand.landmark))) < min(list(
        map(lambda l: l.x, p2_hand.landmark))) else (p2_hand, p1_hand)

    return p1_hand, p2_hand

#lógica do pedra papel tesoura
def get_winner(p1_move, p2_move):
    if p1_move == p2_move:
        return 0
    elif p1_move == "papel":
        return 1 if p2_move == "pedra" else 2
    elif p1_move == "pedra":
        return 1 if p2_move == "tesoura" else 2
    elif p1_move == "tesoura":
        return 1 if p2_move == "papel" else 2


p1_move = None
p2_move = None
winner = None 
score = [0, 0]

hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    success, img = video.read()

    if not success:
        break
# recebe uma imagem e retorna informações como a posição e orientação das mãos
    position = hands.process(img)
    mhl = position.multi_hand_landmarks
# se nenhuma mao for detectada ou diferente de duas o loop é pulado
    if not mhl or len(mhl) != 2:
        continue

# desenha as linhas e os pontos nas maos
    for hand_landmarks in mhl:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    p1_hand, p2_hand = get_players_hand(mhl)

    p1_next_move = get_hand_gestures(p1_hand)
    p2_next_move = get_hand_gestures(p2_hand)

    if (p1_next_move != p1_move or p2_next_move != p2_move):
        p1_move = p1_next_move
        p2_move = p2_next_move

        winner = get_winner(
            p1_move, p2_move)

        if winner == 1:
            score[0] += 1
        elif winner == 2:
            score[1] += 1

        round_result = "Empate" if winner == 0 else f"Jogador {winner} venceu"
        print(f"{p1_move} x {p2_move} = {round_result}")

#-------------------------------EXIBIÇÃO DE RESULTADOS NA TELA---------------------------------------        

    textoResultado = f"{score[0]} x {score[1]}"
    tamanhoResultado, _ = cv2.getTextSize(textoResultado, cv2.FONT_HERSHEY, 5, 3)
    cv2.putText(img, textoResultado, [(img.shape[1] - tamanhoResultado[0]) // 1, 50], cv2.FONT_HERSHEY,
                2, (255, 0, 0), 6)

    resultado, _ = cv2.getTextSize(round_result, cv2.FONT_HERSHEY, 2, 2)
    cv2.putText(img, round_result, [(img.shape[1] - resultado[0]) // 2, img.shape[0] - resultado[1]], cv2.FONT_HERSHEY_COMPLEX,
                2, (255, 0, 0), 2)


    first_player_size, _ = cv2.getTextSize("Jogador 1", cv2.FONT_HERSHEY_COMPLEX, 2, 2)
    cv2.putText(img, "Jogador 1", (50, img.shape[0] // 2 - first_player_size[1]),
                cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
    cv2.putText(img, p1_move, (50, img.shape[0] // 2 - first_player_size[1] + 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)
 
    second_player_size, _ = cv2.getTextSize(
        "Jogador 2", cv2.FONT_HERSHEY_COMPLEX, 2, 2)
    cv2.putText(img, "Jogador 2", (img.shape[1] - second_player_size[0], img.shape[0] // 2 - second_player_size[1]),
                cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 0), 2)
    cv2.putText(img, p2_move, (img.shape[1] - second_player_size[0], img.shape[0] // 2 - second_player_size[1] + 70),
                cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

    cv2.namedWindow('CP02', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CP02', 960, 540)
    cv2.imshow('CP02', img)
    cv2.waitKey('q')

video.release()
cv2.destroyAllWindows()