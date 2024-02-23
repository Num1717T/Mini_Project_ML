import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# ขนาดของแผ่นยาง
rubber_sheet_width = 1024
rubber_sheet_height = 768
shoe_width = 200
shoe_height = 200

# โหลดภาพของวัตถุที่ต้องการวาด
object_image = Image.open("shoe.png")  # แก้ไขตำแหน่งและชื่อไฟล์ตามที่จริง

# ปรับขนาดของภาพให้เท่ากับขนาดของรองเท้า
object_image_resized = object_image.resize((shoe_width, shoe_height))

# จำนวนรองเท้าที่ต้องการวาด
pop_size = 20
mutation_rate = 0.5

# สร้างโครโมโซม
def create_chromosome():
    return [[
        random.randint(0, rubber_sheet_width - shoe_width),
        random.randint(0, rubber_sheet_height - shoe_height),
        random.uniform(0, 360)
    ] for _ in range(pop_size)]

# คำนวณค่า fitness
def calculate_fitness(chromosomes):
    object_image_np = np.array(object_image)
    total_area = rubber_sheet_width * rubber_sheet_height
    intersection_area = 0
    white_area = 0
    fitness = 0
    overlap_penalty = 0

    for i in range(len(chromosomes)):
        shoe_x, shoe_y, rotation_angle = chromosomes[i]

        # คำนวณวงรีขอบเขตหลังการหมุน
        bounding_ellipse = (
            shoe_x + shoe_width / 2, shoe_y + shoe_height / 2,
            shoe_width - 120, shoe_height,
            rotation_angle
        )

        # วาดวงรีขอบแบบหมุนบนรูปภาพ
        cv2.ellipse(object_image_np, (int(bounding_ellipse[0]), int(bounding_ellipse[1])),
                    (int(bounding_ellipse[2]), int(bounding_ellipse[3])), bounding_ellipse[4], 0, 360, (0, 255, 0), -1)

        for j in range(i + 1, len(chromosomes)):
            # คำนวณวงรีขอบเขตสำหรับรองเท้าอันที่สอง
            x_other, y_other, rotation_other = chromosomes[j]
            bounding_ellipse_other = (
                x_other + shoe_width / 2, y_other + shoe_height / 2,
                shoe_width - 120, shoe_height,
                rotation_other
            )

            # ตรวจสอบการทับซ้อนกันระหว่างวงรีที่มีขอบเขต
            overlap = check_ellipse_overlap(bounding_ellipse, bounding_ellipse_other)

            if overlap:
                overlap_penalty -= 1.5

    # นับพื้นที่สีขาวของภาพ
    white_area = np.sum(object_image_np == 255)

    # คำนวณพื้นที่ที่รองเท้าหุ้มอยู่
    intersection_area = total_area - np.sum(object_image_np == 0)

    # คำนวณสมรรถภาพโดยพิจารณาจากพื้นที่ที่รองเท้าหุ้มอยู่ พื้นที่สีขาวทั้งหมด และค่าปรับสำหรับการทับซ้อนกัน
    fitness = (intersection_area / white_area) + overlap_penalty

    return fitness
def check_ellipse_overlap(ellipse1, ellipse2):
    # ตรวจสอบการทับซ้อนกันระหว่างวงรีที่มีขอบเขต
    distance_squared = (ellipse1[0] - ellipse2[0]) ** 2 + (ellipse1[1] - ellipse2[1]) ** 2
    radii_sum_squared = (ellipse1[2] + ellipse2[2]) ** 2

    return distance_squared < radii_sum_squared

# การครอสโอเวอร์ (Crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(0, pop_size-1)
    return [parent1[0:crossover_point] + parent2[crossover_point:],
            parent2[0:crossover_point] + parent1[crossover_point:]]

# การมิวเทชัน (Mutation)
def mutation(chromosome):
    mutated_chromosome = chromosome[:]

    # สุ่มเลขความน่าจะเป็น
    probability = random.uniform(0, 1)

    # เลือกตำแหน่งที่จะมีการเปลี่ยนแปลงของรองเท้าเป็นแบบสุ่ม
    index_to_mutate = random.randint(0, pop_size - 1)

    # ถ้าความน่าจะเป็นน้อยกว่า 0.5 ให้สุ่มตำแหน่งใหม่ของรองเท้า
    if probability < mutation_rate:
        mutated_chromosome[index_to_mutate] = [random.randint(0, rubber_sheet_width - shoe_width),
                                               random.randint(0, rubber_sheet_height - shoe_height),
                                               random.uniform(0, 360)]
    else:
        # ถ้าความน่าจะเป็นมากกว่าหรือเท่ากับ 0.5 ให้สลับตำแหน่งของรองเท้าที่ถูกเลือกกับตำแหน่งอื่น
        other_index = random.randint(0, pop_size - 1)
        mutated_chromosome[index_to_mutate], mutated_chromosome[other_index] = mutated_chromosome[other_index], \
            mutated_chromosome[index_to_mutate]

    # ปรับค่าให้สอดคล้องกับขอบของแผ่นยาง
    shoe_x, shoe_y, rotate = mutated_chromosome[index_to_mutate]
    if shoe_x + shoe_width > rubber_sheet_width:
        mutated_chromosome[index_to_mutate][0] = rubber_sheet_width - shoe_width
    if shoe_y + shoe_height > rubber_sheet_height:
        mutated_chromosome[index_to_mutate][1] = rubber_sheet_height - shoe_height

    return mutated_chromosome

# การตัดสินใจ (Selection)
def selection(population, fitness_values):
    sorted_population = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    return sorted_population[:2]  # เลือก 2 แบบที่ดีที่สุด

# การรัน GA
def run_genetic_algorithm(num_generations):
    population_size = pop_size
    population = [create_chromosome() for _ in range(population_size)]

    for generation in range(num_generations):
        new_population = []

        # คำนวณค่า Fitness สำหรับทุกๆ รอบของ GA
        fitness_values = [calculate_fitness(chromosome) for chromosome in population]

        for _ in range(population_size):
            # การเลือกผู้ปกครอง
            parent = selection(population, fitness_values)

            # การทำ Crossover
            offspring1, offspring2 = crossover(parent[0], parent[1])

            # การทำ Mutation
            offspring1 = mutation(offspring1)
            offspring2 = mutation(offspring2)
            new_population.extend([offspring1, offspring2])
        population = new_population

        # เลือกผลลัพธ์ที่ดีที่สุดหลังจากการรัน GA
        best_solution = max(population, key=calculate_fitness)

        # พิมพ์ผลลัพธ์ Fitness หลังจากทุกรอบของ GA
        best_fitness = calculate_fitness(best_solution)
        print(f"Generation {generation + 1}, Fitness: {best_fitness}")

    return best_solution

# ฟังก์ชันสำหรับทดลองปรับจำนวนรอบการเรียกใช้ GA
def experiment(num_generations):
    best_solution = run_genetic_algorithm(num_generations)
    print("Best Solution:", best_solution)
    print("Fitness:", calculate_fitness(best_solution))

    # แสดงผลลัพธ์
    plt.figure(figsize=(10, 6))
    plt.title('Rubber Sheet with Object Inserts')
    plt.xlim(0, rubber_sheet_width)
    plt.ylim(0, rubber_sheet_height)
    plt.gca().add_patch(plt.Rectangle((0, 0), rubber_sheet_width, rubber_sheet_height, fill=None, edgecolor='red'))

    for shoe in best_solution:
        shoe_x, shoe_y, rotation_angle = shoe

        # หมุนภาพวัตถุเพื่อแสดงภาพ
        rotated_object = object_image_resized.rotate(rotation_angle)
        plt.imshow(rotated_object, extent=[shoe_x, shoe_x + shoe_width, shoe_y, shoe_y + shoe_height], alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.grid(True)
    plt.show()

# ทดลองเรียกใช้งาน GA กับจำนวนรอบการทดสอบแต่ละครั้ง
print("Experiment with 500 generations:")
experiment(500)
