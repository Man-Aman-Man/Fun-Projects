import cv2
import numpy as np
import pygame
import sys

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2-Player Webcam Bat Game")

# Colors
WHITE = (255, 255, 255)
BALL_COLOR = (0, 255, 0)
PADDLE1_COLOR = (0, 255, 0)   # Green for Player 1
PADDLE2_COLOR = (0, 0, 255)   # Blue for Player 2

# Paddle and Ball settings
paddle_width, paddle_height = 100, 10
ball_radius = 10
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 4, 4

# Paddle Y positions
paddle1_y = HEIGHT - 30  # Bottom (Player 1)
paddle2_y = 20           # Top (Player 2)

# Scores
score1, score2 = 0, 0
font = pygame.font.SysFont(None, 36)

# Webcam Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam could not be opened.")
    sys.exit()

def get_paddle_position(frame, color='green'):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == 'green':
        lower = np.array([40, 70, 70])
        upper = np.array([80, 255, 255])
    elif color == 'blue':
        lower = np.array([100, 150, 50])
        upper = np.array([140, 255, 255])
    else:
        return WIDTH // 2

    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(largest)
        center_x = x + w // 2
        return center_x
    return WIDTH // 2

clock = pygame.time.Clock()

# Game loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break
    frame = cv2.flip(frame, 1)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            sys.exit()

    # Get paddle X positions
    paddle1_x = get_paddle_position(frame, 'green') - paddle_width // 2
    paddle2_x = get_paddle_position(frame, 'blue') - paddle_width // 2

    # Update ball position
    ball_x += ball_dx
    ball_y += ball_dy

    # Ball collision with left/right walls
    if ball_x <= 0 or ball_x >= WIDTH:
        ball_dx *= -1

    # Ball collision with Player 1 paddle
    if paddle1_y <= ball_y + ball_radius <= paddle1_y + paddle_height and paddle1_x <= ball_x <= paddle1_x + paddle_width:
        ball_dy *= -1

    # Ball collision with Player 2 paddle
    if paddle2_y <= ball_y - ball_radius <= paddle2_y + paddle_height and paddle2_x <= ball_x <= paddle2_x + paddle_width:
        ball_dy *= -1

    # Ball goes out of bounds (update score and reset)
    if ball_y > HEIGHT:
        score2 += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dy *= -1
    elif ball_y < 0:
        score1 += 1
        ball_x, ball_y = WIDTH // 2, HEIGHT // 2
        ball_dy *= -1

    # Draw game
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, PADDLE1_COLOR, (paddle1_x, paddle1_y, paddle_width, paddle_height))  # Player 1
    pygame.draw.rect(screen, PADDLE2_COLOR, (paddle2_x, paddle2_y, paddle_width, paddle_height))  # Player 2
    pygame.draw.circle(screen, BALL_COLOR, (ball_x, ball_y), ball_radius)

    # Display scores
    score_text1 = font.render(f"Player 1: {score1}", True, WHITE)
    score_text2 = font.render(f"Player 2: {score2}", True, WHITE)
    screen.blit(score_text1, (10, HEIGHT - 40))
    screen.blit(score_text2, (10, 10))

    pygame.display.flip()
    clock.tick(60)

    # Optional webcam preview (press 'q' to quit)
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up on exit
cap.release()
cv2.destroyAllWindows()
pygame.quit() 
