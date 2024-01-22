import numpy as np
import matplotlib.pyplot as plt


def crc_generator(massiv):
    G = [1, 0, 1, 1, 1, 1, 0, 1]
    massiv += [0] * 7  # добавление 7 нулей
    for i in range(len(massiv) - 8):
        if massiv[i] == 1:  # пропуск если 0
            for j in range(8):
                massiv[i + j] ^= G[j]  # xor

    return massiv[-7:]  # оставляем последнии 7 бит


def gold_generator(G):
    x = [0, 1, 1, 0, 1]
    y = [1, 0, 0, 1, 0]
    arr = []
    for i in range(G):
        arr.append(x[4] ^ y[4])
        xx = x[3] ^ x[4]
        del x[-1]
        x.insert(0, xx)

        yy = y[1] ^ y[4]
        del y[-1]  # удаляем последний элемент
        y.insert(0, yy)  # уставляем в начало из сумматора

    return list(arr)


def cor_priem(arr, gold):
    gold_x = np.repeat(gold, 10)

    q = np.correlate(arr, gold_x)
    w = np.argmax(q)

    return arr[w:]


def decod(arr):
    d_bit = []
    for i in range(0, len(arr), 10):
        average = int(sum(arr[i : i + 10])) / 10
        if average > 0.5:
            d_bit.append(1)
        else:
            d_bit.append(0)

    return d_bit


def crc_proverka(massiv):
    G = [1, 0, 1, 1, 1, 1, 0, 1]
    for i in range(len(massiv) - 8):
        if massiv[i] == 1:  # пропуск если 0
            for j in range(8):
                massiv[i + j] ^= G[j]  # xor

    return massiv[-7:]  # оставляем последнии 7 бит


first_name = input("Введите имя: ")
last_name = input("Введите фамилию: ")

# Кодирование в ASCII
cod_str = "".join(
    format(ord(char), "08b") for char in first_name + " " + last_name
)  # добавил пробел чтобы было нормально
bit_p = np.array([int(bit) for bit in cod_str])

plt.figure(1, figsize=(12, 5))
plt.subplots_adjust(hspace=1)
plt.subplot(311)
plt.title("Битовая  последовательность")
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(bit_p)
print(bit_p)
# plt.show()

crc = crc_generator(list(bit_p))
print("CRC = {}".format(crc))

posl_Gold = gold_generator(31)
print("Gold = {}".format(posl_Gold))

Nx = list(posl_Gold) + list(bit_p) + list(crc)
Nx13 = Nx
Nx = list(posl_Gold) + list(bit_p) + list(crc)

plt.subplot(312)
plt.title("Модуляция")
plt.xlabel("Биты")
plt.ylabel("Амплитуда")
plt.plot(Nx)

Nx_2x = [0] * (len(Nx) * 2)

q = input("Введите значение 0 - {}: ".format(len(Nx)))
q = int(q)
Nx_2x[q : q + len(Nx)] = Nx

plt.subplot(313)
plt.title("Модуляция 2x")
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(Nx_2x)

noice = np.random.normal(0, 0.4, len(Nx_2x))
Nx_2x_n = Nx_2x + noice
plt.figure(2, figsize=(12, 5))
plt.subplots_adjust(hspace=1)
plt.subplot(211)
plt.title("Сигнал с шумом")
plt.xlabel("Время")
plt.ylabel("Амплитуда")
plt.plot(Nx_2x_n)

signal = cor_priem(Nx_2x_n, posl_Gold)

plt.subplot(212)
plt.title("Сигнал с шумом начиная с Gold")
plt.xlabel("Время")
plt.ylabel("Амплитуда")
plt.plot(signal)

r = decod(signal)

plt.figure(3, figsize=(12, 5))
plt.subplots_adjust(hspace=1)
plt.subplot(311)
plt.title("Декодированные биты")
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(r)

r = r[: len(posl_Gold) + len(bit_p) + len(crc)]  # Убираем лишнее в конце

plt.subplot(312)
plt.title("Биты без лишнего в конце")
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(r)

r = r[len(posl_Gold) :]

plt.subplot(313)
plt.title("Биты без Gold")
plt.xlabel("Биты")
plt.ylabel("Значение")
plt.plot(r)

crc = crc_proverka(r.copy())
if all(x == 0 for x in crc):
    print("Удачно")
else:
    print("Неудачно")

r = r[:-7]
print(r, sep="")
r = np.array(r)

str = "".join(
    [chr(int("".join(map(str, byte)), 2)) for byte in np.array_split(r, len(r) // 8)]
)
print(str, "\n")

# 13
fft1 = abs(np.fft.fftshift(np.fft.fft(Nx_2x))) / len(Nx_2x) + 0.1
fft2 = abs(np.fft.fftshift(np.fft.fft(Nx_2x_n))) / len(Nx_2x_n)
x = np.arange(-len(fft1) / 2, len(fft1) / 2)
plt.figure(4)
plt.title("Переданый и полученный")
plt.xlabel("Частота [Гц]")
plt.ylabel("Амплитуда")
plt.plot(x, fft1)
plt.plot(x, fft2)

# Разный repeat
fft05 = np.repeat(Nx13, 5)
fft1 = np.repeat(Nx13, 10)
fft2 = np.repeat(Nx13, 20)

# делаем длинну одинаковой
fft05 = list(fft05) + list(fft05)
fft2 = fft2[: len(fft1)]

x = np.arange(-len(fft1) / 2, len(fft1) / 2)

# делим для нормирования
fft05 = np.fft.fft(fft05) / len(fft05) 
fft1 = np.fft.fft(fft1) / len(fft1) 
fft2 = np.fft.fft(fft2) / len(fft2)

# Зануление первого элемента(для красоты)
fft05[0] = 0
fft1[0] = 0
fft2[0] = 0

fft05 = abs(np.fft.fftshift(fft05)) + 0.2
fft1 = abs(np.fft.fftshift(fft1)) + 0.1
fft2 = abs(np.fft.fftshift(fft2))

plt.figure(5)
plt.title("Разные repeat")
plt.xlabel("Частота [Гц]")
plt.ylabel("Амплитуда")
plt.plot(x, fft05)
plt.plot(x, fft1)
plt.plot(x, fft2)

plt.show()