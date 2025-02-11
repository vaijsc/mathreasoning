import re
import torch
# Test the function
test_cases = ["Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\n#### 10", "10 cans are holding 8 liters each for a total of 10 * 8 = <<10*8=80>>80 liters\nThere are 290 - 80 = <<290-80=210>>210 litres left\nThere are 24 - 10 =<<24-10=14>>14 cans left\nEach of the remaining cans is holding 210 / 14 = <<210/14=15>>15 liters each\n#### 15", "The solid wall is 8 ft. * 20 ft. = <<8*20=160>>160 sq. ft.\nThe doorway is 3 ft. * 7 ft. = <<3*7=21>>21 sq. ft.\nThe window is 6 ft. * 4 ft. = <<6*4=24>>24 sq. ft.\nThe closet door is 5 ft. * 7 ft. = <<5*7=35>>35 sq. ft.\nThe total area of the doors and windows is 21 sq. ft + 24 sq. ft. + 35 sq. ft. = <<21+24+35=80>>80 sq. ft.\nThe solid wall is 160 sq. ft., so before the areas of the doors and window are taken into account, the total wall area is 4 * 160 sq. ft. = <<4*160=640>>640 sq. ft.\nTaking into account the doors and window, the total wall area Linda will have to paint is 640 sq. ft. - 80 sq. ft. = <<640-80=560>>560 sq. ft.\n#### 560", "360 + 198 = 123\n12374 * 12312 = 1231", "There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.\nSo in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.\nSo in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.\n#### 35", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\n Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72","She works 8 hours a day for $18 per hour so she makes 8*18 = $<<8*18=144.00>>144.00 per 8-hour shift\nShe works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = <<10-8=2>>2 hours of overtime\nOvertime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = $<<18*.5=9.00>>9.00\nHer overtime pay is 18+9 = $<<18+9=27.00>>27.00\nHer base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = $<<144*5=720.00>>720.00\nHer overtime pay is $27.00 per hour and she works 2 hours of overtime per day and makes 27*2 = $<<27*2=54.00>>54.00 in overtime pay\n2 hours of overtime pay for 5 days means she makes 54*5 = $270.00\nIn 5 days her base pay is $720.00 and she makes $270.00 in overtime pay so she makes $720 + $270 = $<<720+270=990.00>>990.00\n#### 990", "He sold each DVD for 6*2.5=$<<6*2.5=15>>15\nSo he makes a profit of 15-6=$<<15-6=9>>9\nSo each day he makes a profit of 9*500=$<<9*500=4500>>4500\nSo he makes 4500*5=$<<4500*5=22500>>22,500\nHe makes 22,500*20=$<<22500*20=450000>>450,000\nThen after the cost of creating the movie he has a profit of 450,000-2000=$<<450000-2000=448000>>448,000\n#### 448000"]

# sentence_pattern = r"(?:\s*\d+.*|\s*[A-Z].*|#+.*)(?:[.?!](?!\d)|[\n]*|(?=$))"
sentence_pattern = r"\s*(?:\d+|[A-Z]|#+).*?(?:[.?!](?!\d)|[\n]|(?=$))"
equation_pattern = r"((?:\s+\$*\d+\s*)(?:.*?)<<(?:.*?)>>(?:\S*\d+)|\s*#+.*(?=$))"
# equation_pattern = r"((?:\s+\S*\d+\s*)(?:.*?)<<(?:.*?)>>(?:\S*\d+)|\s*#+.*(?=$))"

for sentence in test_cases:
    print(re.findall(equation_pattern, sentence))

def fill_mask(original_seq, rethink_seq):
    pattern = r"<sentinel_tok_(\d+)>(.*?)(?=<sentinel_tok_(\d+)>)"
    single_sentinel_pattern = "(<sentinel_tok_{sentinel_idx}>)"

    fillers = re.findall(pattern=pattern, string=rethink_text)
    for filler in fillers:
        original_seq = re.sub(pattern=single_sentinel_pattern.format(sentinel_idx=filler[0]), repl=f" {filler[1]} ", string=original_seq)
        original_seq = re.sub(r"  ", r" ", original_seq)
    return original_seq

rethink_text = "<sentinel_tok_1>some text<sentinel_tok_2>and<sentinel_tok_3>other text<sentinel_tok_4>"
original_text = "This is<sentinel_tok_1><sentinel_tok_2>some<sentinel_tok_3>"

print(fill_mask(original_text, rethink_text))
answer_pattern = r"#+\s*([-]?\d+(?:[.,]\d*)*)"
test_answer = """He will have 3 + 5 = <<3+5=8>>8 clients this week.                               
  He will use 8 * 2 = <<8*2=16>>16 bottles of bleach.                               
  He will use 8 * 5 = <<8*5=40>>40 packs of cloths.                               
  He will spend 16 * $2 = $<<16*2=32>>32 on bleach.                               
  He will spend 40 * $5 = $<<40*5=200>>200 on cloths.                               
  His total expenses will be $32 + $200 = $<<32+200=232>>232.                               
  His profit will be $92 - $232 = $<<92-232=-140>>-140.                               
  #### -140.5"""

test_answer2 = """Rissa was given $5 x 3 = $<<5*3=15>>15.                                                                                                                                        
  They each gave $15/2 = $<<15/2=7.50>>7.50 to their little sister.                                                                                                                           
  So, Rissa is left with $7.50 x 4/5 = $<<7.5*4/5=6>>6.                                                                                                                                       
  Therefore, Ronnie is left with $5 - $7.50 = $<<5-7.5=-2.50>>-2.50.                                                                                                                          
  #### -2.5"""

test_string = " .\nEliza's earnings for the remaining 5 hours is $10 x 1.2 x 5 = $<<10*1.2*5=60>>60 hours is $400 + $60 = $<<400+60=460>>460 .\nTherefore, Eliza's total earnings for this week is $460.every day at the farmers' market. \n#### 18 "

print(re.findall(pattern=r"\s*(?:\d+|[A-Z]|#+).*?(?:[.?!](?!\d)|[\n]|(?=$))", string=test_answer2))

pattern = r"sentinel_tok_(\d+)>((.|\n)*?)(?=<sentinel_tok_(\d+)>)"
print(re.findall(pattern=pattern, string="<sentinel_tok_1> 16 - 3 - 4 = <<16-3-4=9>>9<sentinel_tok_1> 9 x $2 = $<<9*2=18>>18<sentinel_tok_2>\n#### 18<sentinel_tok_3>"))
# test_case_id = -1 
# for match in re.finditer(equation_pattern, test_cases[test_case_id]):
#     print(test_cases[test_case_id][match.start(): match.end()])
# print(re.findall(equation_pattern, test_cases[test_case_id]))

# def union(idx1, idx2, parents:list, heights:list):
#     par1 = find(idx1, parents)
#     par2 = find(idx2, parents)
#     
#     if par1 == par2:
#         return
# 
#     if heights[par1] >= heights[par2]:
#         parents[idx2] = par1
#         heights[idx2] += 1 
#     elif heights[par1] < heights[par2]:
#         parents[idx1] = par2
#         heights[idx2] += 1
# 
# def find(idx1, parents):
#     if parents[idx1] == idx1:
#         return idx1
# 
#     cur = idx1
#     while cur != parents[cur]:
#         cur = parents[cur]
# 
#     parents[idx1] = cur
#     return cur
# 
# def overlap(obj1:list, obj2:list):
#     if obj1[1] < obj2[0] or obj2[1] < obj1[0]:
#         return False
#     return True
# 
# def disjoint_masked_indices(masked_objs):
#     heights = [0 for _ in range(len(masked_objs))]
#     parents = [i for i in range(len(masked_objs))]
#     
#     adjacent = {i: [] for i in range(len(masked_objs))}
#     for i in range(len(masked_objs)):
#         for j in range(i+1, len(masked_objs)):
#             if overlap(masked_objs[j], masked_objs[i]):
#                 adjacent[i].append(j)
#                 adjacent[j].append(i)
# 
#     for i in range(len(masked_objs)):
#         for j in adjacent[i]:
#             union(idx1=i, idx2=j, parents=parents, heights=heights)
# 
#     groups = {i: None for i in set(parents)}
# 
#     for obj_idx, obj in enumerate(masked_objs):
#         par = find(obj_idx, parents=parents)
#         if groups[par] is None:
#             groups[par] = obj
#         else:
#             groups[par] = [min(obj[0], groups[par][0]), max(obj[1], groups[par][1])]
#    
#     return sorted(list(groups.values()), key=lambda x: x[0])
# 
# test = [[1, 5], [7, 9], [8, 10], [2, 3], [3, 9]]
# test = [[7, 9], [8, 10], [1, 5], [2, 3]]
# print(disjoint_masked_indices(test))
def prepare_ul2_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", prefix_lengths, dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.
    
    Not yet supporting sequence packing
    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    device = attention_mask_with_indices.device
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long)).to(device)
    
    # bidirectional prefix 
    for batch in range(attention_mask_4d.shape[0]):
        for i in range(attention_mask_4d.shape[-1]):
            if attention_mask_4d[batch, :, i, i] == 1:
                break
        # print("hellooo")
        # print(batch, i, prefix_lengths[batch].item())
        prefix_len = int(prefix_lengths[batch].item())
        attention_mask_4d[batch, :, i:i+prefix_len, i:i+prefix_len] = 1.0
    
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d

print(prepare_ul2_4d_attention_mask(torch.Tensor([[1, 1, 1, 1]]), prefix_lengths=torch.Tensor([2]), dtype=torch.float16))

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Abel-7B-002")

conversation = [{"role": "user", "content": "Hello, What is it?"}]

chat = tok.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

print(chat)

