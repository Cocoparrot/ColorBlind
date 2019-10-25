import pygame



# initialize pygame and setup the window and variables we gonna need later
pygame.init()
Main_window = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Colorblind correction')
#Put in the image we want to daltonize
surface = pygame.image.load("Image.jpg").convert()


#Green weak
def deuternopia(surface=pygame.Surface((1, 1))):
    pixel = pygame.Color(0, 0, 0)
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            pixel = surface.get_at((x, y))
            surface.set_at((x, y), pygame.Color(int(pixel.r * 0.75), int(pixel.g * 0.2), int(pixel.b * 0.8)))

#Red weak
def protonapia(surface=pygame.Surface((1,1))):
    pixel = pygame.Color(0, 0, 0)
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            pixel = surface.get_at((x, y))
            surface.set_at((x, y), pygame.Color(int(pixel.r * 0.4), int(pixel.g * 0.8), int(pixel.b * 0.6)))

def tritanopia(surface=pygame.Surface((1,1))):
    pixel = pygame.Color(0, 0, 0)
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            pixel = surface.get_at((x, y))
            surface.set_at((x, y), pygame.Color(int(pixel.r * 0.7), int(pixel.g * 0.35), int(pixel.b * 0.4)))

#protonapia(surface)
#deuternopia(surface)
tritanopia(surface)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    Main_window.fill((255, 255, 255))
    Main_window.blit(surface, (0, 0))
    pygame.display.update()

pygame.quit()