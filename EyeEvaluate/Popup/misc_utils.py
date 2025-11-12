from GlobalData import *


#------处理换行-------------
def wrap_message(message):
    """
    将 message 文本处理为每行不超过 max_length_per_line 字符，保留原有换行，
    并尽可能少的行数。
    """
    return "\n".join(
        line[i:i+max_length_per_line]
        for line in message.split("\n")
        for i in range(0, len(line), max_length_per_line)
    )

def wrap_title(message):
    """
    将 message 文本处理为每行不超过 max_length_per_line 字符，保留原有换行，
    并尽可能少的行数。
    """

    max_length_per_title = max_length_per_line - 6
    return "\n".join(
        line[i:i+max_length_per_title]
        for line in message.split("\n")
        for i in range(0, len(line), max_length_per_title)
    )