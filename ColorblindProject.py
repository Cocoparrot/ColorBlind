import pygame



# initialize pygame and setup the window and variables we gonna need later
pygame.init()
Main_window = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Colorblind correction')
#Put in the image we want to daltonize
image = pygame.image.load("Image.jpg").convert()
renew = pygame.image.load("Image.jpg").convert()
start_count = pygame.time.get_ticks()

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


#Blue weak
def tritanopia(surface=pygame.Surface((1,1))):
    pixel = pygame.Color(0, 0, 0)
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            pixel = surface.get_at((x, y))
            surface.set_at((x, y), pygame.Color(int(pixel.r * 0.7), int(pixel.g * 0.35), int(pixel.b * 0.4)))


running = True
while running:
    for event in pygame.event.get():
        #getting the keys that are pressed
        keys = pygame.key.get_pressed()
        if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
            running = False
        elif event.type == pygame.KEYDOWN:
            if keys[pygame.K_1]:
                Main_window.blit(renew, (0,0))
                pygame.display.update()
                deuternopia(image)
                Main_window.blit(image, (0, 0))
                pygame.display.update()
            if keys[pygame.K_2]:
                Main_window.blit(renew, (0, 0))
                pygame.display.update()
                tritanopia(image)
                Main_window.blit(image, (0, 0))
                pygame.display.update()
            if keys[pygame.K_3]:
                Main_window.blit(renew, (0, 0))
                pygame.display.update()
                protonapia(image)
                Main_window.blit(image, (0, 0))
                pygame.display.update()
        Main_window.fill((255, 255, 255))
        Main_window.blit(image, (0, 0))
        pygame.display.update()


pygame.quit()